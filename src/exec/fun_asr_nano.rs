//! Fun-ASR-Nano ASR driver. Usage: tiny-cpm asr funasr <model-dir> <audio-file> [max_tokens]
//!
//! Ported from aha (github.com/jhqxxx/aha) src/exec/fun_asr_nano.rs, with the model
//! loading from src/models/fun_asr_nano/generate.rs and the decode loop modeled on
//! aha's rocket-free `generate_generic_text` (src/models/common/generate.rs).
//! Transcript goes to stdout; all diagnostics go to stderr.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, Tensor, pickle::read_all_with_key};
use candle_nn::VarBuilder;

use crate::common::modules::sdpa_fast_guard;
use crate::common::sample::{get_logit_processor, use_repeat_penalty};
use crate::common::{InferenceModel, MultiModalData};
use crate::models::fun_asr_nano::{
    config::FunASRNanoConfig, model::FunAsrNanoModel, processor::FunAsrNanoProcessor,
};
use crate::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use crate::tokenizer::TokenizerModel;
use crate::utils::audio_utils::load_audio_with_resample;

/// aha's default seed (mes.seed.unwrap_or(299792458)).
const SEED: u64 = 299792458;
/// aha's GenerationContext default repeat_last_n.
const REPEAT_LAST_N: usize = 64;
/// Default ASR prompt (Fun-ASR convention; aha's tests use the same text).
const DEFAULT_PROMPT: &str = "语音转写：";

/// Inlined from aha src/utils/mod.rs (tiny-cpm's shared utils don't carry it).
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

/// aha's get_dtype, Metal-only variant: bf16 and f16 are both fine on Metal.
fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => match cfg_dtype {
            "float32" | "float" => DType::F32,
            "float64" | "double" => DType::F64,
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            "uint8" => DType::U8,
            "int8" | "int16" | "int32" | "int64" => DType::I64,
            _ => DType::F32,
        },
    }
}

/// aha's FunAsrNanoGenerateModel (generate.rs), minus the rocket response types.
/// Reused as the ASR stage of the `dialogue` pipeline.
pub struct FunAsrEngine {
    tokenizer: TokenizerModel,
    processor: FunAsrNanoProcessor,
    model: FunAsrNanoModel,
    device: Device,
    dtype: DType,
    sample_rate: usize,
    generation_config: Qwen3GenerationConfig,
}

impl FunAsrEngine {
    pub fn load(path: &str, device: &Device) -> Result<Self> {
        let llm_config_path = path.to_string() + "/Qwen3-0.6B";
        let tokenizer = TokenizerModel::init(&llm_config_path)?;
        let generation_config_path = llm_config_path.clone() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let config_path = llm_config_path + "/config.json";
        let llm_cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let config_path = path.to_string() + "/config.yaml";
        let cfg: FunASRNanoConfig = serde_yaml::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = cfg.llm_conf.llm_dtype.as_str();
        let dtype = get_dtype(None, cfg_dtype);
        let processor = FunAsrNanoProcessor::new(&cfg.frontend_conf, device)?;
        let model_list = find_type_files(path, "pt")?;
        if model_list.is_empty() {
            bail!("no .pt weight files found in {path}");
        }
        let mut dict_to_hashmap = HashMap::new();
        for m in model_list {
            let dict = match read_all_with_key(m.clone(), Some("state_dict")) {
                Ok(dict) => dict,
                Err(e) => {
                    eprintln!(
                        "model read_all_with_key {m} get state_dict err: {e}, use None try again"
                    );
                    match read_all_with_key(m.clone(), None) {
                        Ok(dict) => dict,
                        Err(e) => {
                            return Err(anyhow!(format!(
                                "model read_all_with_key({}, None): e: {}",
                                &m, e
                            )));
                        }
                    }
                }
            };
            for (k, v) in dict {
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, device);
        let model =
            FunAsrNanoModel::new(vb, &cfg, &llm_cfg, generation_config.eos_token_id.clone())?;
        Ok(Self {
            tokenizer,
            processor,
            model,
            device: device.clone(),
            dtype,
            sample_rate: cfg.frontend_conf.fs,
            generation_config,
        })
    }

    /// Decode loop modeled on aha's `generate_generic_text`.
    pub fn transcribe(&mut self, audio_path: &str, max_tokens: usize) -> Result<String> {
        let fs = self.sample_rate;
        let audio = load_audio_with_resample(audio_path, &self.device, Some(fs), Some(1))?;
        let n_samples = audio.dim(1)?;
        eprintln!(
            "audio: {} samples ({:.2}s @ {}Hz)",
            n_samples,
            n_samples as f64 / fs as f64,
            fs
        );
        let (speech, fbank_mask, input_ids) =
            self.processor
                .process_info(DEFAULT_PROMPT, &audio, &self.tokenizer)?;
        let speech = speech.to_dtype(self.dtype)?;
        let data = MultiModalData::new(vec![speech.into(), fbank_mask.into()]);

        let mut logit_processor = get_logit_processor(
            Some(self.generation_config.temperature),
            Some(self.generation_config.top_p),
            Some(self.generation_config.top_k),
            SEED,
        );
        let repeat_penalty = self.generation_config.repetition_penalty;
        let prompt_len = input_ids.dim(1)?;
        eprintln!("prompt tokens: {prompt_len}");

        let mut generated: Vec<u32> = Vec::new();
        let eos_ids = self.model.stop_token_ids();
        let mut seqlen_offset = 0usize;
        let mut seq_len = prompt_len;

        // Route the Qwen3-0.6B decode through the fused SDPA path (same guard the
        // identical-backbone Qwen3-ASR uses in qwen3_asr.rs): skips repeat_kv + its
        // contiguous copy per token. The is_decode check inside
        // eager_attention_forward auto-falls-back to eager for the seq>1 prefill, so
        // wrapping the whole transcribe is safe. Decode is GPU-forward-bound, and
        // this is exactly the lever that cuts it.
        let _sdpa_fast = sdpa_fast_guard();
        let i_start = Instant::now();
        let logits = self
            .model
            .forward_initial(&input_ids, seqlen_offset, data)?;
        let next_token = sample_and_push(
            &mut logit_processor,
            repeat_penalty,
            &logits,
            &mut generated,
        )?;
        let ttft = i_start.elapsed();
        eprintln!("TTFT (incl. audio encoder + adaptor): {ttft:?}");
        seqlen_offset += seq_len;
        seq_len = 1;
        let mut input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;

        // 自回归循环
        let i_start = Instant::now();
        for _ in 1..max_tokens {
            let logits = self.model.forward_step(&input_ids, seqlen_offset)?;
            let next_token = sample_and_push(
                &mut logit_processor,
                repeat_penalty,
                &logits,
                &mut generated,
            )?;

            if eos_ids.contains(&next_token) {
                break;
            }
            seqlen_offset += seq_len;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }
        let decode_duration = i_start.elapsed();
        self.model.clear_cache();

        let n_decode = generated.len().saturating_sub(1);
        if decode_duration.as_secs_f64() > 0.0 {
            eprintln!(
                "decode: {} tokens in {:?} ({:.1} tok/s)",
                n_decode,
                decode_duration,
                n_decode as f64 / decode_duration.as_secs_f64()
            );
        }
        let text = self.tokenizer.token_decode(generated)?;
        Ok(text)
    }
}

/// aha's sample_and_push: squeeze to rank-1 f32 logits, repeat penalty, sample, push.
fn sample_and_push(
    logit_processor: &mut candle_transformers::generation::LogitsProcessor,
    repeat_penalty: f32,
    logits: &Tensor,
    generated: &mut Vec<u32>,
) -> Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    // 重复惩罚
    let logits = use_repeat_penalty(repeat_penalty, Some(REPEAT_LAST_N), &logits, generated)?;
    let token = logit_processor.sample(&logits)?;
    generated.push(token);
    Ok(token)
}

pub fn run(args: &[String]) -> Result<()> {
    let Some(model_dir) = args.first() else {
        bail!("usage: tiny-cpm asr funasr <model-dir> <audio-file> [max_tokens]");
    };
    let Some(audio_file) = args.get(1) else {
        bail!("usage: tiny-cpm asr funasr <model-dir> <audio-file> [max_tokens]");
    };
    let max_tokens = args
        .get(2)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| anyhow!("invalid max_tokens: {e}"))?
        .unwrap_or(512);

    let device = Device::new_metal(0)?;

    let i_start = Instant::now();
    let mut model = FunAsrEngine::load(model_dir, &device)?;
    eprintln!("Time elapsed in load model is: {:?}", i_start.elapsed());

    let i_start = Instant::now();
    let text = model.transcribe(audio_file, max_tokens)?;
    eprintln!("Time elapsed in generate is: {:?}", i_start.elapsed());

    println!("{text}");
    Ok(())
}
