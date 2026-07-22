//! Qwen3-ASR driver. Usage: tiny-cpm asr qwen3 <model-dir> <audio-file> [max_tokens]
//!
//! Ported from aha (github.com/jhqxxx/aha): model init from
//! src/models/qwen3_asr/generate.rs (`Qwen3AsrGenerateModel::init`), the decode
//! loop from its `generate`/`asr_audio` methods, and CLI shape from
//! src/exec/qwen3_asr.rs. aha's minijinja chat template is replaced by the
//! hard-coded default template string aha itself uses in `asr_audio`.
//!
//! `Qwen3AsrEngine` is the reusable form (also the ASR stage of the `live`
//! subcommand): load once, transcribe files or in-memory 16kHz mono samples.
//! `run()` is a thin wrapper over it with unchanged CLI behavior.

use std::time::Instant;

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    common::sample::get_logit_processor,
    models::{
        feature_extractor::config::FeatureExtractor,
        qwen3_asr::{
            config::{Qwen3ASRConfig, Qwen3ASRGenerationConfig},
            model::Qwen3ASRModel,
            processor::{AudioData, Qwen3AsrProcessor},
        },
    },
    tokenizer::TokenizerModel,
};

/// aha generate.rs `default_template` (used by `asr_audio` in place of the
/// minijinja chat template). The single `<|audio_pad|>` is expanded to the
/// audio feature length by the processor.
const DEFAULT_TEMPLATE: &str = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n";

/// aha src/utils/mod.rs `clean_asr_response` (private copy; shared files are untouched).
///
/// Qwen3ASR outputs format: "language English<asr_text>The morning sun..."
/// This function extracts the text after "<asr_text>" marker.
/// If no marker is found, returns the original text trimmed (for compatibility).
fn clean_asr_response(raw: &str) -> String {
    if let Some(start) = raw.find("<asr_text>") {
        raw[start + "<asr_text>".len()..].trim().to_string()
    } else {
        raw.trim().to_string()
    }
}

/// aha src/utils/mod.rs `get_dtype` (metal branch; bf16 is fine on Apple Metal).
fn get_dtype(cfg_dtype: &str) -> DType {
    match cfg_dtype {
        "float32" | "float" => DType::F32,
        "float64" | "double" => DType::F64,
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        "uint8" => DType::U8,
        "int8" | "int16" | "int32" | "int64" => DType::I64,
        _ => DType::F32,
    }
}

/// aha src/utils/mod.rs `find_type_files`.
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

/// Qwen3-ASR model + processor + tokenizer, loaded once and reusable.
/// Also serves as the ASR stage of the `live` subcommand.
pub struct Qwen3AsrEngine {
    device: Device,
    tokenizer: TokenizerModel,
    processor: Qwen3AsrProcessor,
    model: Qwen3ASRModel,
    dtype: DType,
    /// Number of safetensors shards (for the load diagnostics line).
    pub(crate) n_shards: usize,
    eos_token_id1: u32,
    eos_token_id2: u32,
    temperature: f32,
}

impl Qwen3AsrEngine {
    /// aha Qwen3AsrGenerateModel::init: tokenizer + configs + safetensors.
    pub fn load(model_dir: &str, device: &Device) -> Result<Self> {
        let tokenizer = TokenizerModel::init(model_dir)?;
        let generation_config_path = model_dir.to_string() + "/generation_config.json";
        let generation_config: Qwen3ASRGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        if generation_config.eos_token_id.len() < 2 {
            bail!("generation_config.json: expected 2 eos_token_id entries");
        }
        let preprocess_config_path = model_dir.to_string() + "/preprocessor_config.json";
        let preprocess_config: FeatureExtractor =
            serde_json::from_slice(&std::fs::read(preprocess_config_path)?)?;
        let processor = Qwen3AsrProcessor::new(device, &preprocess_config)?;
        let config_path = model_dir.to_string() + "/config.json";
        let cfg: Qwen3ASRConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let dtype = get_dtype(cfg.thinker_config.dtype.as_str());
        let model_list = find_type_files(model_dir, "safetensors")?;
        if model_list.is_empty() {
            bail!("no safetensors files found in {model_dir}");
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let model = Qwen3ASRModel::new(vb, &cfg, generation_config.eos_token_id.clone())?;
        Ok(Self {
            device: device.clone(),
            tokenizer,
            processor,
            model,
            dtype,
            n_shards: model_list.len(),
            eos_token_id1: generation_config.eos_token_id[0],
            eos_token_id2: generation_config.eos_token_id[1],
            temperature: generation_config.temperature,
        })
    }

    /// Model dtype (bf16 on Metal), for diagnostics.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// aha generate.rs `generate` (non-stream), minus chat params: temperature
    /// and seed come from generation_config.json / aha's fixed default.
    /// Returns the cleaned transcript plus (chunks, prompt tokens, generated
    /// tokens, decode wall time) for diagnostics.
    fn decode_audio_datas(
        &mut self,
        audio_datas: &[AudioData],
        max_tokens: usize,
    ) -> Result<(String, usize, usize, usize, std::time::Duration)> {
        let seed = 34562u64;
        let mut logit_processor = get_logit_processor(Some(self.temperature), None, None, seed);
        let i_start = Instant::now();
        let mut generate: Vec<u32> = Vec::new();
        let mut prompt_tokens = 0usize;
        for data in audio_datas.iter() {
            let mut input_ids = data.input_ids.clone();
            let mut input_features = Some(data.input_features.clone().to_dtype(self.dtype)?);
            let mut seq_len = input_ids.dim(1)?;
            prompt_tokens += seq_len;
            let mut seqlen_offset = 0;
            for _ in 0..max_tokens {
                let logits =
                    self.model
                        .forward(&input_ids, seqlen_offset, input_features.as_ref())?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
                generate.push(next_token);
                if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                input_features = None;
            }
            self.model.clear_kv_cache();
        }
        let elapsed = i_start.elapsed();
        let text = self.tokenizer.token_decode(generate.clone())?;
        let text = clean_asr_response(&text);
        Ok((
            text,
            audio_datas.len(),
            prompt_tokens,
            generate.len(),
            elapsed,
        ))
    }

    /// Transcribe an audio file (any format symphonia reads; resampled to
    /// 16kHz mono internally). Mirrors aha's `asr_audio` path.
    pub fn transcribe_path(&mut self, audio_file: &str, max_tokens: usize) -> Result<String> {
        let audio_datas =
            self.processor
                .process_audio_path(DEFAULT_TEMPLATE, audio_file, &self.tokenizer)?;
        let (text, chunks, prompt_tokens, generated, elapsed) =
            self.decode_audio_datas(&audio_datas, max_tokens)?;
        eprintln!(
            "{} chunk(s), {} prompt tokens, {} generated tokens in {:?} ({:.2} tok/s)",
            chunks,
            prompt_tokens,
            generated,
            elapsed,
            generated as f64 / elapsed.as_secs_f64()
        );
        Ok(text)
    }

    /// Transcribe in-memory 16kHz mono f32 samples (e.g. a VAD segment from
    /// the `live` loop). Same mel/template/decode pipeline as the file path,
    /// minus loading and long-audio chunking (segments are far under the
    /// 1200s single-chunk limit).
    pub fn transcribe_samples(
        &mut self,
        samples_16k_mono: &[f32],
        max_tokens: usize,
    ) -> Result<String> {
        let audio = Tensor::new(samples_16k_mono, &self.device)?;
        let data =
            self.processor
                .process_audio_tensor(DEFAULT_TEMPLATE, &audio, &self.tokenizer)?;
        let (text, _, _, _, _) = self.decode_audio_datas(&[data], max_tokens)?;
        Ok(text)
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let Some(model_dir) = args.first() else {
        bail!("usage: tiny-cpm asr qwen3 <model-dir> <audio-file> [max_tokens]");
    };
    let Some(audio_file) = args.get(1) else {
        bail!("usage: tiny-cpm asr qwen3 <model-dir> <audio-file> [max_tokens]");
    };
    let max_tokens: usize = match args.get(2) {
        Some(s) => s
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid max_tokens: {s}"))?,
        None => 512, // same default as aha's asr_audio
    };

    let device = Device::new_metal(0)?;

    let i_start = Instant::now();
    let mut engine = Qwen3AsrEngine::load(model_dir, &device)?;
    eprintln!(
        "loaded model in {:?} ({} safetensors shard(s), dtype {:?})",
        i_start.elapsed(),
        engine.n_shards,
        engine.dtype()
    );

    let text = engine.transcribe_path(audio_file, max_tokens)?;
    println!("{text}");
    Ok(())
}
