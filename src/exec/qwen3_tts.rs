//! Qwen3-TTS 12Hz 1.7B Base driver.
//!   tiny-cpm tts qwen3 <model-dir> "<text>" <out.wav> [--ref <ref.wav> --ref-text "<text>"] [--language <lang>] [--max-frames N]
//! Ported from github.com/QwenLM/Qwen3-TTS (the `qwen_tts` package); the loading and
//! synthesis flow mirrors `qwen_tts/inference/qwen3_tts_model.py::generate_voice_clone`.
//!
//! Weights: `model.safetensors` (BF16 — talker + code predictor + speaker encoder) and
//! `speech_tokenizer/model.safetensors` (F32 — the 12 Hz codec). Text tokenizer is the
//! Qwen2 BPE (`vocab.json` + `merges.txt`). Without `--ref` the model synthesizes in its
//! default voice; `--ref` + `--ref-text` enable ICL zero-shot voice cloning (the speaker
//! encoder embeds the ref clip and the codec encoder turns it into prompt codes).

use std::time::Instant;

use anyhow::{Result, anyhow};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::models::qwen3_tts::codec::SpeechTokenizer;
use crate::models::qwen3_tts::config::{
    Qwen3TTSConfig, Qwen3TTSGenerationConfig, SpeakerEncoderParams, SpeechTokenizerConfig,
};
use crate::models::qwen3_tts::speaker_encoder::SpeakerEncoder;
use crate::models::qwen3_tts::talker::{RefVoice, Talker};
use crate::tokenizer::TokenizerModel;
use crate::utils::audio_utils::{load_audio_with_resample, save_wav_mono};

const USAGE: &str = "usage: tiny-cpm tts qwen3 <model-dir> \"<text>\" <out.wav> [--ref <ref.wav> --ref-text \"<text>\"] [--language <lang>] [--max-frames N] [--talker-quant <q4_k|q8_0|none>] [--stream] [--stream-first N] [--stream-chunk N]";

/// Default first chunk (frames @ 12.5 Hz) before the first audio flush: 12 frames ≈
/// 0.96 s of audio — small enough for a fast first-audio, large enough to not spend the
/// whole budget on overlapping codec windows. Env-tunable via QWEN3_TTS_STREAM_FIRST.
const STREAM_FIRST_FRAMES: usize = 12;
/// Steady-state chunk between flushes (25 frames = 2.0 s of audio). QWEN3_TTS_STREAM_CHUNK.
const STREAM_CHUNK_FRAMES: usize = 25;
/// Left context re-decoded (and dropped) per flush so chunk seams match the batch path.
/// Matches the batch `chunked_decode(.., 25)` left context. QWEN3_TTS_STREAM_CTX.
const STREAM_LEFT_CONTEXT: usize = 25;

/// Talker backbone precision: QMatMul (Q4_K/Q8_0, runtime-quantized in memory)
/// or the full BF16 safetensors path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TalkerQuant {
    Q4K,
    Q8_0,
    /// Diagnostic: run the QuantizedTalkerLayer code with FULL-precision (F32)
    /// weights — isolates "layer-logic bug" from "quantization loss". If this
    /// is clean but q4_k/q8_0 babble, the bug is quantization numerics, not code.
    PassthroughF32,
    None,
}

impl TalkerQuant {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "q4k" | "q4km" => Ok(Self::Q4K),
            "q80" => Ok(Self::Q8_0),
            "passthrough" | "passthroughf32" | "f32quant" => Ok(Self::PassthroughF32),
            "none" | "bf16" | "f16" | "full" => Ok(Self::None),
            other => Err(anyhow!(
                "unknown --talker-quant `{other}` (expected q4_k | q8_0 | none)"
            )),
        }
    }
    fn ggml(self) -> Option<GgmlDType> {
        match self {
            Self::Q4K => Some(GgmlDType::Q4K),
            Self::Q8_0 => Some(GgmlDType::Q8_0),
            Self::PassthroughF32 => Some(GgmlDType::F32),
            Self::None => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Q4K => "q4_k",
            Self::Q8_0 => "q8_0",
            Self::PassthroughF32 => "passthrough-f32",
            Self::None => "bf16",
        }
    }
}

/// aha src/utils/mod.rs `find_type_files` (same as exec/qwen3_asr.rs).
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
    files.sort();
    Ok(files)
}

/// Read a usize env knob with a default (used for the streaming chunk tunables).
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

pub struct Qwen3TtsEngine {
    device: Device,
    talker: Talker,
    codec: SpeechTokenizer,
    speaker_encoder: SpeakerEncoder,
    tokenizer: TokenizerModel,
    gen_cfg: Qwen3TTSGenerationConfig,
    sample_rate: usize,
    talker_dtype: DType,
}

impl Qwen3TtsEngine {
    pub fn load(model_dir: &str, device: &Device) -> Result<Self> {
        // Default BF16 (fastest on Metal; see the --talker-quant note in `run`).
        Self::load_with_quant(model_dir, device, TalkerQuant::None)
    }

    pub fn load_with_quant(model_dir: &str, device: &Device, quant: TalkerQuant) -> Result<Self> {
        // --- main config + generation defaults ---
        let tts_cfg: Qwen3TTSConfig =
            serde_json::from_slice(&std::fs::read(format!("{model_dir}/config.json"))?)?;
        let gen_path = format!("{model_dir}/generation_config.json");
        let gen_cfg: Qwen3TTSGenerationConfig = if std::path::Path::new(&gen_path).exists() {
            serde_json::from_slice(&std::fs::read(gen_path)?)?
        } else {
            Qwen3TTSGenerationConfig::default()
        };

        // --- talker + code predictor + speaker encoder (BF16 safetensors) ---
        let dtype = match std::env::var("TINY_CPM_QWEN3_TTS_DTYPE").as_deref() {
            Ok("f32") | Ok("float32") => DType::F32,
            _ => DType::BF16, // proven on Metal by Qwen3-ASR
        };
        let model_list = find_type_files(model_dir, "safetensors")?;
        if model_list.is_empty() {
            return Err(anyhow!("no safetensors found in {model_dir}"));
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let talker = match quant.ggml() {
            Some(ggml) => {
                // Runtime in-memory quantization: mmap the talker weights on CPU,
                // quantize the 7 big matmuls/layer onto Metal (quantize_onto needs
                // a CPU source). The rest of the talker loads from `vb` (Metal).
                let tq = Instant::now();
                let vb_cpu = unsafe {
                    VarBuilder::from_mmaped_safetensors(&model_list, dtype, &Device::Cpu)?
                };
                let talker = Talker::new_quantized(vb.clone(), vb_cpu, &tts_cfg, ggml, device)?;
                eprintln!(
                    "qwen3-tts: talker backbone = {} (quantized in {:.2?})",
                    quant.label(),
                    tq.elapsed()
                );
                talker
            }
            None => {
                eprintln!("qwen3-tts: talker backbone = bf16");
                Talker::new(vb.clone(), &tts_cfg, device)?
            }
        };
        // Speaker encoder stays F32: its STFT/mel front-end is F32 and the ECAPA convs
        // are small (76 tensors) — matches the official F32 reference and avoids an
        // F32-mel → BF16-conv dtype mismatch. Its output embedding is cast to the talker
        // dtype in `encode_ref`.
        let vb_f32 =
            unsafe { VarBuilder::from_mmaped_safetensors(&model_list, DType::F32, device)? };
        // enc_dim comes from config.json's speaker_encoder_config (1024 on 0.6B, 2048 on
        // 1.7B); the rest of the ECAPA-TDNN hyper-params are the upstream defaults.
        let spk_params = SpeakerEncoderParams {
            enc_dim: tts_cfg.speaker_encoder_config.enc_dim,
            sample_rate: tts_cfg.speaker_encoder_config.sample_rate,
            ..SpeakerEncoderParams::default()
        };
        let speaker_encoder = SpeakerEncoder::new(
            vb_f32.pp("speaker_encoder"),
            spk_params,
            device,
        )?;

        // --- 12 Hz codec (F32 safetensors in speech_tokenizer/) ---
        let codec_dir = format!("{model_dir}/speech_tokenizer");
        let codec_cfg: SpeechTokenizerConfig =
            serde_json::from_slice(&std::fs::read(format!("{codec_dir}/config.json"))?)?;
        let codec_list = find_type_files(&codec_dir, "safetensors")?;
        if codec_list.is_empty() {
            return Err(anyhow!("no safetensors found in {codec_dir}"));
        }
        let codec_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&codec_list, DType::F32, device)? };
        let sample_rate = codec_cfg.output_sample_rate;
        let codec = SpeechTokenizer::new(codec_vb, &codec_cfg, true)?;

        let tokenizer = TokenizerModel::init(model_dir)?;
        Ok(Self {
            device: device.clone(),
            talker,
            codec,
            speaker_encoder,
            tokenizer,
            gen_cfg,
            sample_rate,
            talker_dtype: dtype,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Tokenize plain text into the talker's target-text ids (no chat wrapper — the
    /// talker builds the role prefix itself).
    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.text_encode_vec(text.to_string(), false)
    }

    /// The BPE id for '\n' used in the "<|im_start|>assistant\n" role prefix.
    fn newline_token(&self) -> Result<u32> {
        let ids = self.tokenizer.text_encode_vec("\n".to_string(), false)?;
        ids.first()
            .copied()
            .ok_or_else(|| anyhow!("tokenizer: '\\n' encodes to nothing"))
    }

    /// Pre-encode a reference voice (speaker embedding + ref codec codes + ref text ids).
    pub fn encode_ref(&self, ref_wav: &str, ref_text: &str) -> Result<RefVoice> {
        let wav = load_audio_with_resample(ref_wav, &self.device, Some(self.sample_rate), Some(1))?; // (1, T)
        let pcm: Vec<f32> = wav.squeeze(0)?.to_vec1::<f32>()?;
        let spk_embed = self
            .speaker_encoder
            .embed(&pcm)?
            .to_dtype(self.talker_dtype)?; // (enc_dim,)
        // codec encode expects (B, 1, T).
        let ref_code = self
            .codec
            .encoder
            .as_ref()
            .ok_or_else(|| anyhow!("codec encoder not loaded"))?
            .encode(&wav.unsqueeze(1)?)?; // (1, 16, T_ref)
        let ref_code = ref_code.squeeze(0)?; // (16, T_ref)
        let ref_text_ids = self.encode_text(ref_text)?;
        Ok(RefVoice {
            spk_embed,
            ref_code,
            ref_text_ids,
        })
    }

    /// Synthesize `text` to 24 kHz mono PCM (1, T). `ref_voice` enables ICL cloning.
    pub fn synthesize_pcm(
        &mut self,
        text: &str,
        language: &str,
        ref_voice: Option<&RefVoice>,
        max_frames: usize,
    ) -> Result<Tensor> {
        let text_ids = self.encode_text(text)?;
        let newline = self.newline_token()?;
        let codes = self.talker.generate(
            &text_ids,
            language,
            ref_voice,
            newline,
            &self.gen_cfg,
            max_frames,
        )?; // (n_frames, 16)
        // codec decode expects (B, 16, T).
        let codes = codes.t()?.unsqueeze(0)?; // (1, 16, n_frames)
        let wav = self.codec.decoder.chunked_decode(&codes, 300, 25)?; // (1, 1, T*1920)
        Ok(wav.squeeze(0)?) // (1, T*1920)
    }

    /// Streaming synthesis: codec-decode incrementally as the talker emits frames and
    /// fire `on_audio` with each new PCM tail. Returns the full waveform (== what
    /// `synthesize_pcm` produces) plus time-to-first-audio in seconds.
    ///
    /// Unlike cosyvoice3's streaming (which RE-RUNS flow+HiFT over the whole prefix per
    /// chunk), Qwen3-TTS's codec is causal with a left-context trim (`chunked_decode`),
    /// so each flush decodes only a sliding window `[emitted-left_context .. now]` and
    /// emits the new tail — total decode work stays O(n_frames), not O(n²).
    ///
    /// Fidelity note: a flush window's left context (25 frames) does NOT fully cover the
    /// codec's receptive field, so a small chunk approximates the batch PCM rather than
    /// matching it bit-for-bit. The gap is window-size-limited and collapses to 0 as the
    /// chunk reaches the batch's 300-frame window (measured: first=12/chunk=25 → max|Δ|
    /// ≈ 1.2 in PCM units; chunk=300 → 0.0; regression guard `tests/stream_decode_equiv.rs`).
    /// Seams stay continuous (no clicks); the approximation lives near chunk boundaries.
    /// Larger `chunk_frames` = closer to batch but later audio; smaller = faster first
    /// audio but a rougher opening. Tune with `--stream-first` / `--stream-chunk`.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_pcm_streaming(
        &mut self,
        text: &str,
        language: &str,
        ref_voice: Option<&RefVoice>,
        max_frames: usize,
        first_chunk_frames: usize,
        chunk_frames: usize,
        left_context: usize,
        on_audio: &mut dyn FnMut(&[f32]),
    ) -> Result<(Tensor, f64)> {
        use std::cell::RefCell;
        let text_ids = self.encode_text(text)?;
        let newline = self.newline_token()?;
        let frame_samples = self.codec.decoder.frame_samples(); // 1920 @ 24kHz
        let device = self.device.clone();
        let t0 = Instant::now();

        // All streaming state lives in one RefCell so the frame callback can borrow it
        // while `&mut self.talker` drives generation (disjoint fields, no alias).
        struct St {
            frames: Vec<Vec<u32>>, // every generated frame (16 codes)
            emitted: usize,        // frames whose PCM is already emitted
            first_secs: Option<f64>,
        }
        let st = RefCell::new(St {
            frames: Vec::new(),
            emitted: 0,
            first_secs: None,
        });
        // Decode error captured inside the callback (which can't return Result).
        let decode_err: RefCell<Option<anyhow::Error>> = RefCell::new(None);

        // Decode the not-yet-emitted frames as ONE window (left context re-decoded, then
        // dropped) and fire `on_audio` with the new tail. Single `decode` of the window +
        // a manual `ctx * frame_samples` drop == the batch `chunked_decode` math exactly.
        let mut flush = |st: &mut St, codec: &SpeechTokenizer, force: bool| {
            let threshold = if st.emitted == 0 {
                first_chunk_frames
            } else {
                chunk_frames
            };
            let avail = st.frames.len() - st.emitted;
            if avail == 0 || (!force && avail < threshold) {
                return;
            }
            let win_start = st.emitted.saturating_sub(left_context);
            let ctx = st.emitted - win_start; // left-context frames re-decoded & dropped
            let n_win = st.frames.len() - win_start;
            let flat: Vec<u32> = st.frames[win_start..]
                .iter()
                .flat_map(|f| f.iter().copied())
                .collect();
            let wav: Result<Vec<f32>> = (|| {
                let codes = Tensor::from_vec(flat, (1, 16, n_win), &device)?;
                let w = codec.decoder.decode(&codes)?; // (1, 1, n_win*1920)
                Ok(w.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?)
            })();
            let wav = match wav {
                Ok(v) => v,
                Err(e) => {
                    *decode_err.borrow_mut() = Some(e.into());
                    return;
                }
            };
            // Keep only the tail past the already-emitted region (context + prev frames).
            let skip = ctx * frame_samples;
            let new_tail = &wav[skip.min(wav.len())..];
            if st.first_secs.is_none() && !new_tail.is_empty() {
                st.first_secs = Some(t0.elapsed().as_secs_f64());
            }
            if !new_tail.is_empty() {
                on_audio(new_tail);
            }
            st.emitted = st.frames.len();
        };

        let codec_ref = &self.codec;
        let mut gen_err: Option<anyhow::Error> = None;
        {
            let mut on_frame = |frame: &[u32]| {
                let mut s = st.borrow_mut();
                s.frames.push(frame.to_vec());
                let threshold = if s.emitted == 0 {
                    first_chunk_frames
                } else {
                    chunk_frames
                };
                if s.frames.len() - s.emitted >= threshold {
                    flush(&mut s, codec_ref, false);
                }
                // Stop generating once a decode error was recorded; it surfaces after.
                decode_err.borrow().is_none()
            };
            if let Err(e) = self.talker.generate_stream(
                &text_ids,
                language,
                ref_voice,
                newline,
                &self.gen_cfg,
                max_frames,
                Some(&mut on_frame),
            ) {
                gen_err = Some(e);
            }
        }
        if let Some(e) = decode_err.borrow_mut().take() {
            return Err(e);
        }
        if let Some(e) = gen_err {
            return Err(e);
        }
        // Final flush of the remaining partial chunk.
        {
            let mut s = st.borrow_mut();
            flush(&mut s, codec_ref, true);
        }
        if let Some(e) = decode_err.borrow_mut().take() {
            return Err(e);
        }

        let st = st.into_inner();
        // Re-decode the full waveform for the return value so the caller saves the exact
        // batch-reference WAV (the streamed chunks are identical; this just reuses the
        // standard layout). Cheap relative to generation.
        let flat: Vec<u32> = st.frames.iter().flat_map(|f| f.iter().copied()).collect();
        let n = st.frames.len();
        let codes = Tensor::from_vec(flat, (n, 16), &self.device)?;
        let codes = codes.t()?.unsqueeze(0)?;
        let wav = self.codec.decoder.chunked_decode(&codes, 300, 25)?;
        Ok((wav.squeeze(0)?, st.first_secs.unwrap_or(f64::NAN)))
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        out_wav: &str,
        language: &str,
        ref_voice: Option<&RefVoice>,
        max_frames: usize,
    ) -> Result<()> {
        let pcm = self.synthesize_pcm(text, language, ref_voice, max_frames)?;
        save_wav_mono(&pcm, out_wav, self.sample_rate as u32)?;
        Ok(())
    }

    /// Codec roundtrip (verification): encode a wav to 16-codebook codes, decode back to
    /// a wav. Validates the encoder + decoder RVQ/convs independent of the talker.
    pub fn codec_roundtrip(&self, in_wav: &str, out_wav: &str) -> Result<()> {
        let wav = load_audio_with_resample(in_wav, &self.device, Some(self.sample_rate), Some(1))?; // (1, T)
        let codes = self
            .codec
            .encoder
            .as_ref()
            .ok_or_else(|| anyhow!("codec encoder not loaded"))?
            .encode(&wav.unsqueeze(1)?)?; // (1, 16, T_frames)
        eprintln!(
            "qwen3-tts: codec roundtrip codes shape = {:?}",
            codes.shape()
        );
        let pcm = self.codec.decoder.chunked_decode(&codes, 300, 25)?; // (1, 1, T*1920)
        save_wav_mono(&pcm.squeeze(0)?, out_wav, self.sample_rate as u32)?;
        Ok(())
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let mut positional: Vec<&str> = Vec::new();
    let mut ref_wav: Option<String> = None;
    let mut ref_text: Option<String> = None;
    let mut language = "auto".to_string();
    let mut max_frames: usize = 2048; // generation_config.json max_new_tokens (wrapper)
    let mut codec_roundtrip = false; // hidden verification path
    let mut stream = false; // chunked streaming synthesis
    let mut stream_first = env_usize("QWEN3_TTS_STREAM_FIRST", STREAM_FIRST_FRAMES);
    let mut stream_chunk = env_usize("QWEN3_TTS_STREAM_CHUNK", STREAM_CHUNK_FRAMES);
    // Talker backbone precision: CLI flag > env > default q4_k.
    let mut talker_quant: Option<TalkerQuant> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ref" => {
                i += 1;
                ref_wav = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--ref requires a <ref.wav> path"))?
                        .clone(),
                );
            }
            "--ref-text" => {
                i += 1;
                ref_text = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--ref-text requires a value"))?
                        .clone(),
                );
            }
            "--language" => {
                i += 1;
                language = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--language requires a value"))?
                    .clone();
            }
            "--max-frames" => {
                i += 1;
                max_frames = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-frames requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--max-frames must be a positive integer"))?;
            }
            "--talker-quant" => {
                i += 1;
                talker_quant =
                    Some(TalkerQuant::parse(args.get(i).ok_or_else(|| {
                        anyhow!("--talker-quant requires a value")
                    })?)?);
            }
            "--codec-roundtrip" => {
                codec_roundtrip = true;
            }
            "--stream" => {
                stream = true;
            }
            "--stream-first" => {
                i += 1;
                stream_first = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--stream-first requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--stream-first must be a positive integer"))?;
            }
            "--stream-chunk" => {
                i += 1;
                stream_chunk = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--stream-chunk requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--stream-chunk must be a positive integer"))?;
            }
            other => positional.push(other),
        }
        i += 1;
    }
    let talker_quant = match talker_quant {
        Some(q) => q,
        None => match std::env::var("TINY_CPM_QWEN3_TTS_TALKER").as_deref() {
            Ok(s) => TalkerQuant::parse(s)?,
            // Default BF16: on Apple Silicon the BF16 matmul is faster than the
            // Q4_K/Q8_0 Metal kernels at decode (T=1) — quantization here saves
            // memory, not speed. Opt in with --talker-quant q4_k/q8_0.
            Err(_) => TalkerQuant::None,
        },
    };
    let [model_dir, text, out_wav] = positional.as_slice() else {
        return Err(anyhow!(USAGE));
    };
    if ref_wav.is_some() && ref_text.is_none() {
        return Err(anyhow!(
            "--ref requires --ref-text (the transcript of the reference wav)"
        ));
    }
    if ref_wav.is_none() && ref_text.is_some() {
        return Err(anyhow!("--ref-text only makes sense together with --ref"));
    }

    // TINY_CPM_DEVICE=cpu forces CPU inference (default: Metal).
    let device = if std::env::var("TINY_CPM_DEVICE").as_deref() == Ok("cpu") {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    eprintln!("device: {device:?}");

    let t0 = Instant::now();
    let mut engine = Qwen3TtsEngine::load_with_quant(model_dir, &device, talker_quant)?;
    eprintln!("qwen3-tts: engine loaded in {:.2?}", t0.elapsed());

    // Hidden verification path: `tts qwen3 <model-dir> <in.wav> <out.wav> --codec-roundtrip`.
    if codec_roundtrip {
        let t0 = Instant::now();
        engine.codec_roundtrip(text, out_wav)?;
        eprintln!(
            "qwen3-tts: codec roundtrip done in {:.2?} → {out_wav}",
            t0.elapsed()
        );
        return Ok(());
    }

    let ref_voice = match &ref_wav {
        Some(path) => {
            let t0 = Instant::now();
            let rv = engine.encode_ref(path, ref_text.as_deref().unwrap_or(""))?;
            eprintln!("qwen3-tts: ref voice encoded in {:.2?}", t0.elapsed());
            Some(rv)
        }
        None => None,
    };

    let t0 = Instant::now();
    if stream {
        let sample_rate = engine.sample_rate(); // hoist: closure borrows engine immutably
        let mut chunk_idx = 0usize;
        let mut chunk_samples: Vec<usize> = Vec::new();
        // Optional self-check (QWEN3_TTS_STREAM_CHECK=1): accumulate the streamed PCM and
        // diff it against the returned batch-reference WAV. The diff is the window-size
        // approximation (see synthesize_pcm_streaming's fidelity note): it should shrink
        // toward 0 as --stream-chunk grows, and hit 0 at chunk=300.
        let check = std::env::var("QWEN3_TTS_STREAM_CHECK").is_ok();
        let mut streamed: Vec<f32> = Vec::new();
        let (pcm, first_secs) = engine.synthesize_pcm_streaming(
            text,
            &language,
            ref_voice.as_ref(),
            max_frames,
            stream_first,
            stream_chunk,
            STREAM_LEFT_CONTEXT,
            &mut |tail: &[f32]| {
                chunk_samples.push(tail.len());
                if check {
                    streamed.extend_from_slice(tail);
                }
                eprintln!(
                    "qwen3-tts: stream chunk {chunk_idx}: +{} samples ({:.2}s audio), {:.2}s elapsed",
                    tail.len(),
                    tail.len() as f64 / sample_rate as f64,
                    t0.elapsed().as_secs_f64()
                );
                chunk_idx += 1;
            },
        )?;
        if check {
            let reference = pcm.squeeze(0)?.to_vec1::<f32>()?;
            let m = streamed.len().min(reference.len());
            let max_diff = streamed[..m]
                .iter()
                .zip(&reference[..m])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "qwen3-tts: stream check: streamed {} vs batch {} samples, max|Δ| = {max_diff:.6} (len Δ = {})",
                streamed.len(),
                reference.len(),
                streamed.len() as isize - reference.len() as isize
            );
        }
        let synth = t0.elapsed();
        let n_samples = pcm.dim(1)?;
        let secs = n_samples as f64 / engine.sample_rate() as f64;
        save_wav_mono(&pcm, out_wav, engine.sample_rate() as u32)?;
        let rtf = synth.as_secs_f64() / secs.max(1e-9);
        eprintln!(
            "qwen3-tts: stream: first audio at {:.2}s ({} chunks), synthesized {secs:.2}s in {synth:.2?} (RTF {rtf:.2}) → {out_wav}",
            first_secs,
            chunk_samples.len()
        );
        return Ok(());
    }

    let pcm = engine.synthesize_pcm(text, &language, ref_voice.as_ref(), max_frames)?;
    let synth = t0.elapsed();
    let n_samples = pcm.dim(1)?;
    let secs = n_samples as f64 / engine.sample_rate() as f64;
    save_wav_mono(&pcm, out_wav, engine.sample_rate() as u32)?;
    let rtf = synth.as_secs_f64() / secs.max(1e-9);
    eprintln!("qwen3-tts: synthesized {secs:.2}s audio in {synth:.2?} (RTF {rtf:.2}) → {out_wav}");
    Ok(())
}
