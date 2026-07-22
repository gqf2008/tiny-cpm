//! MOSS-TTS-Nano TTS driver. Usage: tiny-cpm tts moss <model-dir> "<text>" <out.wav> [--codec <codec-dir>] [--ref <ref.wav>] [--max-len N]
//! Ported from aha (github.com/jhqxxx/aha) src/models/moss_tts_nano/generate.rs
//! (aha has no exec driver for this model; the loading pipeline mirrors its
//! `MossTTSGenerate::init`, the run flow mirrors `tests/test_moss_tts.rs`).
//!
//! - `--codec`: MOSS-Audio-Tokenizer-Nano directory; defaults to
//!   `<model-dir>/../MOSS-Audio-Tokenizer-Nano` (the two are separate HF repos).
//! - `--ref`: reference wav, enables VoiceClone mode; without it the model runs
//!   in Continuation mode (text only).
//! - `--max-len`: cap on generated codec frames. Default 100, matching aha's
//!   hardcoded `sample_len` (MOSS-Audio-Tokenizer-Nano emits 25 fps at 24 kHz,
//!   so the default yields ~4 s of audio).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, pickle::read_all_with_key};
use candle_nn::VarBuilder;
use sentencepiece::SentencePieceProcessor;

use crate::models::{
    moss_audio_tokenizer_nano::{MossAudioTokenizer, config::MossAudioTokenizerConfig},
    moss_tts_nano::{
        config::MossTTSConfig,
        model::{MossGenStats, MossTTSMode, MossTTSModel},
        processor::MossTTSProcessor,
    },
};

/// Default cap on generated codec frames (aha hardcoded `sample_len = 100`).
const DEFAULT_MAX_FRAMES: usize = 100;

// Inlined from aha src/utils/mod.rs (tiny-cpm does not port aha's utils/mod.rs).

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

/// aha `utils::get_dtype`, non-cuda branch (tiny-cpm is Metal-only:
/// bfloat16 maps to F16, as on aha's metal builds).
fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => match cfg_dtype {
            "float32" | "float" => DType::F32,
            "float64" | "double" => DType::F64,
            "float16" | "bfloat16" => DType::F16, // cpu上bfloat16有问题
            "uint8" => DType::U8,
            "int8" | "int16" | "int32" | "int64" => DType::I64,
            _ => DType::F32,
        },
    }
}

/// MOSS-TTS-Nano + MOSS-Audio-Tokenizer-Nano, loaded once and reusable.
/// Also serves as the TTS stage of the `dialogue` pipeline.
pub struct MossEngine {
    device: Device,
    audio_tokenizer: MossAudioTokenizer,
    text_tokenizer: SentencePieceProcessor,
    processor: MossTTSProcessor,
    model: MossTTSModel,
}

impl MossEngine {
    /// Load the TTS model (`model_dir`, .bin pickles, F32) and the codec
    /// (`codec_dir`, safetensors) onto `device`.
    pub fn load(model_dir: &str, codec_dir: &str, device: &Device) -> Result<Self> {
        let load_start = Instant::now();

        // --- codec (MOSS-Audio-Tokenizer-Nano: safetensors + config.json) ---
        let audio_tokenizer_config_path = codec_dir.to_string() + "/config.json";
        let audio_tokenizer_cfg: MossAudioTokenizerConfig =
            serde_json::from_slice(&std::fs::read(audio_tokenizer_config_path)?)?;
        let model_list = find_type_files(codec_dir, "safetensors")?;
        let audio_dtype = get_dtype(None, &audio_tokenizer_cfg.dtype);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, audio_dtype, device)? };
        let audio_tokenizer = MossAudioTokenizer::new(vb, &audio_tokenizer_cfg)?;

        // --- text tokenizer (sentencepiece) ---
        let text_tokenizer_path = model_dir.to_string() + "/tokenizer.model";
        let text_tokenizer = SentencePieceProcessor::open(text_tokenizer_path)
            .map_err(|e| anyhow!(format!("load bpe.model file error:{}", e)))?;

        // --- TTS config + processor ---
        let tts_cfg_path = model_dir.to_string() + "/config.json";
        let tts_cfg: MossTTSConfig = serde_json::from_slice(&std::fs::read(tts_cfg_path)?)?;
        let processor = MossTTSProcessor::new(
            &tts_cfg,
            audio_tokenizer_cfg.sample_rate,
            audio_tokenizer_cfg.number_channels,
            &text_tokenizer,
        )?;

        // --- TTS weights (.bin PyTorch pickles) ---
        // Official Python reference runs this model in F32 on CPU (config.json
        // "dtype": "float32"); aha/older builds forced F16 on Metal, which visibly
        // degraded audio quality on this ~100M model. Use F32 — it stays realtime.
        let model_list = find_type_files(model_dir, "bin")?;
        let mut dict_to_hashmap = HashMap::new();
        let m_dtype = DType::F32;
        for m in model_list {
            let dict = read_all_with_key(m, None)?;
            for (k, v) in dict {
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb = VarBuilder::from_tensors(dict_to_hashmap, m_dtype, device);
        let model = MossTTSModel::new(vb, &tts_cfg)?;
        eprintln!(
            "loaded MOSS-TTS-Nano ({model_dir}) + codec ({codec_dir}) in {:?}",
            load_start.elapsed()
        );

        Ok(Self {
            device: device.clone(),
            audio_tokenizer,
            text_tokenizer,
            processor,
            model,
        })
    }

    /// Synthesize `text` to `out_path` (mirrors aha MossTTSGenerate::generate).
    /// No `ref_wav` => Continuation (text only); `ref_wav` => VoiceClone.
    pub fn synthesize(
        &mut self,
        text: &str,
        out_path: &str,
        max_len: usize,
        ref_wav: Option<&str>,
    ) -> Result<MossGenStats> {
        let mode = if ref_wav.is_some() {
            None
        } else {
            Some(MossTTSMode::Continuation)
        };
        let mode = self
            .processor
            .resolved_mode(mode, false, ref_wav.is_some())?;
        eprintln!("mode: {mode:?}, max frames: {max_len}");
        let prep_start = Instant::now();
        let input_ids = self.processor.build_inference_input_ids(
            text,
            ref_wav,
            None,
            mode,
            &self.audio_tokenizer,
            &self.text_tokenizer,
            &self.device,
        )?;
        eprintln!(
            "input prep (tokenize{}): {:.3}s",
            if ref_wav.is_some() {
                " + ref-wav codec encode"
            } else {
                ""
            },
            prep_start.elapsed().as_secs_f64()
        );
        let stats = self
            .model
            .generate(&input_ids, &self.audio_tokenizer, max_len, out_path)?;
        eprintln!(
            "TTFT (prefill + first codec frame): {:.3}s",
            stats.ttft.as_secs_f64()
        );
        eprintln!(
            "generated {} codec frames ({:.2}s audio) in {:.2}s ({:.2} frames/s, codec decode {:.2}s) -> {out_path}",
            stats.frames,
            stats.frames as f64 * self.audio_tokenizer.downsample_rate as f64
                / (self.audio_tokenizer.sampling_rate * self.audio_tokenizer.number_channels)
                    as f64,
            stats.total.as_secs_f64(),
            stats.frames as f64 / stats.total.as_secs_f64(),
            stats.codec_decode.as_secs_f64(),
        );
        Ok(stats)
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let usage = "usage: tiny-cpm tts moss <model-dir> \"<text>\" <out.wav> [--codec <codec-dir>] [--ref <ref.wav>] [--max-len N]";
    if args.len() < 3 {
        return Err(anyhow!(usage));
    }
    let tts_path = args[0].clone();
    let text = args[1].clone();
    let out_path = args[2].clone();
    let mut codec_path: Option<String> = None;
    let mut ref_wav: Option<String> = None;
    let mut max_len = DEFAULT_MAX_FRAMES;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--codec" => {
                i += 1;
                codec_path = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--codec requires a directory. {usage}"))?
                        .clone(),
                );
            }
            "--ref" => {
                i += 1;
                ref_wav = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--ref requires a wav path. {usage}"))?
                        .clone(),
                );
            }
            "--max-len" => {
                i += 1;
                max_len = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-len requires a frame count. {usage}"))?
                    .parse()
                    .map_err(|_| anyhow!("--max-len must be a positive integer. {usage}"))?;
            }
            other => return Err(anyhow!("unknown option {other}. {usage}")),
        }
        i += 1;
    }
    if max_len == 0 {
        return Err(anyhow!("--max-len must be >= 1"));
    }
    let codec_path = match codec_path {
        Some(p) => p,
        None => {
            let default = Path::new(&tts_path)
                .join("..")
                .join("MOSS-Audio-Tokenizer-Nano");
            if !default.is_dir() {
                return Err(anyhow!(
                    "MOSS-Audio-Tokenizer-Nano codec directory not found at default location {}; pass it explicitly via --codec <codec-dir>",
                    default.display()
                ));
            }
            default.to_string_lossy().to_string()
        }
    };
    if !Path::new(&codec_path).is_dir() {
        return Err(anyhow!("codec directory not found: {codec_path}"));
    }

    // TINY_CPM_DEVICE=cpu forces CPU inference (default: Metal).
    let device = if std::env::var("TINY_CPM_DEVICE").as_deref() == Ok("cpu") {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    eprintln!("device: {device:?}");

    let mut engine = MossEngine::load(&tts_path, &codec_path, &device)?;
    let _stats = engine.synthesize(&text, &out_path, max_len, ref_wav.as_deref())?;
    Ok(())
}
