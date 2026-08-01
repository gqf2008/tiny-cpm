//! Ported from aha (github.com/jhqxxx/aha) src/models/fire_red_vad/vad.rs
//!
//! Adapted: aha's `crate::utils::{find_type_files, get_device}` helpers are not
//! ported to tiny-cpm, so private copies live at the bottom of this file
//! (`get_device` keeps aha's metal-build branch: Metal with Cpu fallback).
use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    common::modules::VadFrameResult,
    models::fire_red_vad::{
        config::{DetectModelConfig, FireRedVadConfig},
        model::DetectModel,
        processor::{AudioFeat, VadPostprocessor},
    },
    utils::{
        audio_utils::{resample_audio_from_bytes, resample_audio_from_vec_f32},
        tensor_utils::split_tensor_with_size,
    },
};

#[derive(Debug)]
pub struct VadResult {
    pub dur: f32,
    pub timestamps: Vec<(f32, f32)>,
    pub model_name: String,
    pub mode: String,
}

/// Optional CLI overrides for the FireRedVAD streaming params that callers
/// (e.g. `live`) want to expose as flags. Each field is `Some` when set on the
/// command line; `None` falls back to the env var, then the realtime default.
/// Priority: CLI flag > env var > default.
#[derive(Default, Clone)]
pub struct VadOverrides {
    pub speech_threshold: Option<f32>,
    pub min_speech_frame: Option<usize>,
    pub min_silence_frame: Option<usize>,
    pub min_speach_ratio: Option<f32>,
    pub end_silence_ratio: Option<f32>,
    pub min_speach_frames: Option<usize>,
    pub look_back_frames: Option<usize>,
}

pub struct FireRedVad {
    audio_feat: AudioFeat,
    vad_model: DetectModel,
    vad_postprocessor: VadPostprocessor,
    model_name: String,
    device: Device,
    cfg: FireRedVadConfig,
    caches: Option<Vec<Tensor>>,
    frame_length_sample: usize,
    speech_cache: Vec<Tensor>,
    pred_cache: Vec<u32>,
    min_speach_frames: usize,
    look_back_frames: usize,
    min_speach_ratio: f32,
    end_silence_ratio: f32,
    // Trailing-silence endpointing for the streaming path: how many CONSECUTIVE
    // non-speech frames must pass before we cut the segment. Without this the
    // else-branch below cut on a SINGLE non-speech frame, so any mid-utterance
    // breath/pause/word-search fragmented one utterance into many short turns
    // ("ASR only hears the first phrase"). This is what the web `silence_ms`
    // slider actually maps onto (silence_ms / 25ms per frame).
    trailing_silence_frames: usize,
    silence_run: usize,
    // Per-frame neural speech flag from the last `detect_frame` call — set from
    // `preds_sum > probs_len * min_speach_ratio` (the same threshold the segment
    // accumulator uses). Exposed for realtime barge-in: a caller can read it to
    // react at speech onset (neural, robust to headphone bleed / breath) instead
    // of a crude RMS threshold.
    last_frame_speech: bool,
}

impl FireRedVad {
    pub fn init(
        path: &str,
        device: Option<&Device>,
        dtype: Option<DType>,
        overrides: Option<VadOverrides>,
    ) -> Result<Self> {
        let device = get_device(device);
        let audio_feat = AudioFeat::new(path, &device)?;
        let model_list = find_type_files(path, "safetensors")?;
        let dtype = dtype.unwrap_or(DType::F32);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("VAD")
            .to_string();
        let (model_cfg, mut cfg) = if model_name.to_lowercase().contains("stream") {
            (
                DetectModelConfig::default_stream_vad(),
                FireRedVadConfig::default_stream_vad(),
            )
        } else if model_name.to_lowercase().contains("aed") {
            (
                DetectModelConfig::default_aed(),
                FireRedVadConfig::default_aed(),
            ) // TODO: aed
        } else {
            (
                DetectModelConfig::default_vad(),
                FireRedVadConfig::default_vad(),
            )
        };
        let vad_model = DetectModel::new(vb, model_cfg)?;
        // CLI overrides (from live flags) take priority over env, which takes
        // priority over the realtime defaults. Helper: cli > env > default.
        let ov = overrides.unwrap_or_default();
        let envf = |key: &str, cli: Option<f32>, default: f32| -> f32 {
            cli.or_else(|| std::env::var(key).ok().and_then(|s| s.parse().ok()))
                .unwrap_or(default)
        };
        let envu = |key: &str, cli: Option<usize>, default: usize| -> usize {
            cli.or_else(|| std::env::var(key).ok().and_then(|s| s.parse().ok()))
                .unwrap_or(default)
        };
        // Postprocessor config (VadPostprocessor reads these).
        cfg.speech_threshold = envf(
            "VAD_SPEECH_THRESHOLD",
            ov.speech_threshold,
            cfg.speech_threshold,
        );
        cfg.min_speech_frame = envu(
            "VAD_MIN_SPEECH_FRAME",
            ov.min_speech_frame,
            cfg.min_speech_frame,
        );
        // Default min_silence_frame 45 (raised from upstream 20) for the
        // whole-file postprocessor path (detect_file; `vad` subcommand). The
        // streaming detect_frame path reads trailing_silence_frames instead —
        // the same env/CLI chain feeds both (see below).
        cfg.min_silence_frame = envu("VAD_MIN_SILENCE_FRAME", ov.min_silence_frame, 45);
        let vad_postprocessor = VadPostprocessor::new(&cfg);
        // Streaming segment logic in detect_frame.
        let min_speach_ratio = envf("VAD_MIN_SPEACH_RATIO", ov.min_speach_ratio, 0.1);
        // Default end_silence_ratio 0.9: require 90% silence in the look-back
        // window. Spiky mics (bone-conduction headsets) produce periodic dips in
        // continuous speech that a lower ratio endpoints on, fragmenting one
        // utterance into many short turns.
        let end_silence_ratio = envf("VAD_END_SILENCE_RATIO", ov.end_silence_ratio, 0.9);
        let min_speach_frames = envu("VAD_MIN_SPEACH_FRAMES", ov.min_speach_frames, 30);
        // Default look_back_frames 50: a longer window needs sustained silence,
        // so inter-phrase dips don't endpoint.
        let look_back_frames = envu("VAD_LOOK_BACK_FRAMES", ov.look_back_frames, 50);
        // Default 32 frames (~800ms at the 25ms frame shift): the streaming
        // detect_frame endpoint only cuts after this many consecutive non-speech
        // frames, matching the web `silence_ms=800` default so mid-utterance
        // pauses up to ~0.8s don't cut the user off. Shares the min_silence_frame
        // env/CLI chain (--vad-min-silence / VAD_MIN_SILENCE_FRAME, in frames),
        // so those knobs now control the streaming path too; update_params keeps
        // it in sync at runtime.
        let trailing_silence_frames = envu("VAD_MIN_SILENCE_FRAME", ov.min_silence_frame, 32);
        eprintln!(
            "vad params: speech_threshold={} min_speech_frame={} min_silence_frame={} trailing_silence_frames={} | min_speach_ratio={} end_silence_ratio={} min_speach_frames={} look_back_frames={}",
            cfg.speech_threshold,
            cfg.min_speech_frame,
            cfg.min_silence_frame,
            trailing_silence_frames,
            min_speach_ratio,
            end_silence_ratio,
            min_speach_frames,
            look_back_frames
        );
        Ok(Self {
            audio_feat,
            vad_model,
            vad_postprocessor,
            model_name,
            device,
            cfg,
            caches: None,
            frame_length_sample: 400,
            speech_cache: vec![],
            pred_cache: vec![],
            min_speach_frames, // ~750ms at 25ms/frame
            look_back_frames,  // ~1.25s at 25ms/frame
            min_speach_ratio,
            end_silence_ratio,
            // Streaming trailing-silence endpoint; default 32 frames (~800ms at
            // the 25ms frame shift), from the min_silence_frame chain above.
            trailing_silence_frames,
            silence_run: 0,
            last_frame_speech: false,
        })
    }

    pub fn detect_frame(&mut self, audio_frame: &Tensor) -> Result<Option<VadFrameResult>> {
        if audio_frame.dim(0)? < self.frame_length_sample {
            return Err(anyhow!(
                "Expected {} samples, got {}",
                self.frame_length_sample,
                audio_frame.dim(0)?
            ));
        }
        let wave_tensor = audio_frame.affine(32768.0, 0.0)?;
        let feats = self.audio_feat.extract(&wave_tensor)?;
        let (probs, caches) = self
            .vad_model
            .forward(&feats.unsqueeze(0)?, self.caches.as_ref())?;
        self.caches = Some(caches);
        let probs = probs.squeeze(D::Minus1)?.squeeze(0)?;
        let binary_preds = self
            .vad_postprocessor
            .process_thresh(&probs)?
            .to_dtype(DType::U32)?;
        let preds_sum = binary_preds.sum_all()?.to_scalar::<u32>()?;
        let probs_len = probs.dim(0)?;
        // 输入数据中 is_speech > 0.1, 认为这帧数据有人声
        let frame_is_speech = preds_sum as f32 > probs_len as f32 * self.min_speach_ratio;
        self.last_frame_speech = frame_is_speech;
        let final_data = if frame_is_speech {
            // Speech: reset the consecutive-silence counter and accumulate speech frames.
            self.silence_run = 0;
            self.speech_cache.push(audio_frame.clone());
            let preds = binary_preds.to_vec1::<u32>()?;
            self.pred_cache.extend_from_slice(&preds);

            // 人声缓存数据过少，等待下一帧。需要同时积累够 min_speach_frames
            // AND 至少 look_back_frames（否则下面 `len - look_back_frames` 在
            // usize 上下溢）；取两者最大值作门槛。
            if self.pred_cache.len() < self.min_speach_frames.max(self.look_back_frames) {
                None
            } else {
                // 判断是否停止说话
                let start = self.pred_cache.len() - self.look_back_frames;
                let look_back = self.pred_cache[start..].iter().sum::<u32>();
                // 判断结尾是否静音
                let silence_ratio = 1.0 - (look_back as f32 / self.look_back_frames as f32);
                if silence_ratio >= self.end_silence_ratio {
                    // 静音返回缓存数据并清空缓存
                    let speech = Tensor::cat(&self.speech_cache, 0)?;
                    self.speech_cache.clear();
                    self.pred_cache.clear();
                    Some(speech)
                } else {
                    // 不是静音此次返回None
                    None
                }
            }
        } else {
            // This frame is non-speech. Don't cut immediately — accumulate
            // consecutive silence and only end the segment once it reaches
            // trailing_silence_frames. That way mid-utterance breaths/pauses/
            // word-searches (brief non-speech frames) don't fragment one
            // utterance into several turns.
            self.silence_run += 1;
            if self.pred_cache.len() >= self.min_speach_frames
                && self.silence_run >= self.trailing_silence_frames
            {
                let data = Tensor::cat(&self.speech_cache, 0)?;
                self.speech_cache.clear();
                self.pred_cache.clear();
                self.silence_run = 0;
                Some(data)
            } else if self.silence_run >= self.trailing_silence_frames {
                // Silence long enough but the speech cache is too short (under
                // min_speach_frames, usually noise): discard.
                self.speech_cache.clear();
                self.pred_cache.clear();
                self.silence_run = 0;
                None
            } else {
                // Still inside the trailing-silence window: keep waiting (cache
                // retained — speech may resume).
                None
            }
        };

        if final_data.is_none() {
            Ok(None)
        } else {
            Ok(Some(VadFrameResult {
                is_speech: true,
                orig_audio: final_data,
                model_name: self.model_name.clone(),
                mode: "speech".to_string(),
            }))
        }
    }

    pub fn detect_frame_f32(
        &mut self,
        audio_vec_f32: Vec<f32>,
        channels: usize,
        orig_sr: Option<usize>,
    ) -> Result<Option<VadFrameResult>> {
        if !self.model_name.to_lowercase().contains("stream") {
            return Err(anyhow!("only stream model support detect_frame"));
        }
        let audio_frame = resample_audio_from_vec_f32(
            audio_vec_f32,
            &self.device,
            channels,
            orig_sr,
            Some(16000),
        )?
        .squeeze(0)?;
        self.detect_frame(&audio_frame)
    }

    pub fn detect_frame_bytes(&mut self, audio_bytes: Vec<u8>) -> Result<Option<VadFrameResult>> {
        if !self.model_name.to_lowercase().contains("stream") {
            return Err(anyhow!("only stream model support detect_frame"));
        }
        let audio_frame =
            resample_audio_from_bytes(audio_bytes, &self.device, Some(16000), 1)?.squeeze(0)?;
        self.detect_frame(&audio_frame)
    }

    pub fn detect_file(&self, audio_path: &str) -> Result<VadResult> {
        let (feats, dur) = self.audio_feat.extract_file(audio_path, &self.device)?;
        let probs = if feats.dim(0)? <= self.cfg.chunk_max_frame {
            let (probs, _) = self.vad_model.forward(&feats.unsqueeze(0)?, None)?;
            probs
        } else {
            let mut chunk_probs = vec![];
            let chunks = split_tensor_with_size(&feats, self.cfg.chunk_max_frame, 0usize)?;
            for chunk in chunks.iter() {
                let (chunk_prob, _) = self.vad_model.forward(&chunk.unsqueeze(0)?, None)?;
                chunk_probs.push(chunk_prob);
            }
            Tensor::cat(&chunk_probs, 1)?
        };
        let probs = if self.model_name.to_lowercase().contains("aed") {
            // only care speech
            probs
                .squeeze(0)?
                .narrow(D::Minus1, 0, 1)?
                .squeeze(D::Minus1)?
        } else {
            probs.squeeze(0)?.squeeze(D::Minus1)?
        };
        let segments = self.vad_postprocessor.process(&probs, dur)?;
        let res = VadResult {
            dur,
            timestamps: segments,
            model_name: self.model_name.clone(),
            mode: "speech".to_string(),
        };
        Ok(res)
    }

    pub fn reset(&mut self) {
        self.caches = None;
        // Clear per-utterance streaming state too, so a reused instance starts a
        // fresh segment (speech/pred caches + the trailing-silence counter).
        self.speech_cache.clear();
        self.pred_cache.clear();
        self.silence_run = 0;
    }

    /// Runtime-update the streaming + postprocessor params (used by the web
    /// UI's live sliders). Each `Some` field is applied; `None` leaves it. The
    /// streaming fields are read per `detect_frame` call; the postprocessor
    /// fields are `pub` on `VadPostprocessor` and read per call too, so changes
    /// take effect on the next frame.
    pub fn update_params(&mut self, o: &VadOverrides) {
        if let Some(v) = o.speech_threshold {
            self.vad_postprocessor.prob_threshold = v;
        }
        if let Some(v) = o.min_speech_frame {
            self.vad_postprocessor.min_speech_frame = v;
        }
        if let Some(v) = o.min_silence_frame {
            self.vad_postprocessor.min_silence_frame = v;
            // Also drive the streaming trailing-silence endpoint: the web
            // `silence_ms` slider is converted to frames (silence_ms/25) by the
            // caller and arrives here as min_silence_frame, so keep the two in
            // sync — this is the value the streaming path actually uses.
            self.trailing_silence_frames = v;
        }
        if let Some(v) = o.min_speach_ratio {
            self.min_speach_ratio = v;
        }
        if let Some(v) = o.end_silence_ratio {
            self.end_silence_ratio = v;
        }
        if let Some(v) = o.min_speach_frames {
            self.min_speach_frames = v;
        }
        if let Some(v) = o.look_back_frames {
            self.look_back_frames = v;
        }
    }

    /// Whether the neural VAD judged the **last** `detect_frame` call's frame as
    /// speech (`preds_sum > probs_len * min_speach_ratio`). Read after
    /// `detect_frame_f32` to drive realtime barge-in at speech onset without a
    /// crude RMS threshold (which can't separate soft speech from headphone
    /// bleed / breath).
    pub fn last_frame_speech(&self) -> bool {
        self.last_frame_speech
    }
}

/// aha `utils::get_device` (metal-build branch; tiny-cpm is Metal-only).
fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => Device::new_metal(0).unwrap_or(Device::Cpu),
    }
}

/// aha `utils::find_type_files`, verbatim.
fn find_type_files(path: &str, extension_type: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.is_file()
            && let Some(extension) = file_path.extension()
            && extension == extension_type
        {
            files.push(file_path.to_string_lossy().to_string());
        }
    }

    Ok(files)
}
