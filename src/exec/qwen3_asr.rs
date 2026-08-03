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

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, Tensor, quantized::GgmlDType};
use candle_nn::VarBuilder;

use crate::{
    common::{modules::sdpa_fast_guard, sample::get_logit_processor},
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
///
/// `#[doc(hidden)] pub`: exposed only so `tests/asr_correctness.rs` can compare
/// the cleaned transcript against a committed golden; not part of the crate API
/// (mirrors the `gpu_sample_token` test-only export in `talker.rs`).
#[doc(hidden)]
pub fn clean_asr_response(raw: &str) -> String {
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

/// Parse the `--quant` / `TINY_CPM_ASR_QUANT` value. `none`/empty → BF16 (default);
/// `q8_0` → `Q8_0` (greedy ASR stays token-exact); `q4_k` → `Q4K` (faster, may flip
/// occasional tokens — measure CER). Same knob set as chat's `TINY_CPM_QUANT`.
fn parse_asr_quant(s: &str) -> Result<Option<GgmlDType>> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "" | "none" => None,
        "q8_0" | "q8" => Some(GgmlDType::Q8_0),
        "q4_k" | "q4k" => Some(GgmlDType::Q4K),
        other => bail!("unknown ASR quant '{other}' (expected q8_0 | q4_k | none)"),
    })
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
    /// generation_config.json `do_sample`. Qwen3-ASR ships `do_sample: false`
    /// (greedy) — honored below so the decode loop uses ArgMax instead of a
    /// temperature-1e-6 multinomial over the full 151936 vocab (which forces a
    /// ~607KB GPU→CPU logits readback + softmax + O(V) CPU draw every token).
    do_sample: bool,
}

impl Qwen3AsrEngine {
    /// aha Qwen3AsrGenerateModel::init: tokenizer + configs + safetensors.
    /// Quant level comes from `TINY_CPM_ASR_QUANT` (none/q8_0/q4_k); `--quant` on
    /// the CLI calls [`Self::load_with_quant`] directly.
    pub fn load(model_dir: &str, device: &Device) -> Result<Self> {
        let quant = match std::env::var("TINY_CPM_ASR_QUANT") {
            Ok(s) => parse_asr_quant(&s)?,
            Err(_) => None,
        };
        Self::load_with_quant(model_dir, device, quant)
    }

    /// Like [`Self::load`] but with an explicit quant level (`None` = BF16). When
    /// `Some`, the 28 thinker decoder layers are quantized in-memory from a
    /// CPU-mmaped copy of the safetensors (`quantize_onto` needs a CPU source); the
    /// audio tower / embeddings / lm_head stay BF16 from the Metal mmap.
    pub fn load_with_quant(
        model_dir: &str,
        device: &Device,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
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
        // kv_cap for the quant path's preallocated KV (the BF16 path's Tensor::cat
        // cache is uncapped). 4096 covers ~300s of audio at ~13 tokens/s; raise via
        // TINY_CPM_ASR_KV_CAP for longer single-chunk audio.
        let kv_cap = std::env::var("TINY_CPM_ASR_KV_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096);
        let model = match quant {
            Some(g) => {
                let tq = Instant::now();
                let vb_cpu = unsafe {
                    VarBuilder::from_mmaped_safetensors(&model_list, dtype, &Device::Cpu)?
                };
                let m = Qwen3ASRModel::new_quantized(
                    vb,
                    vb_cpu,
                    &cfg,
                    generation_config.eos_token_id.clone(),
                    g,
                    kv_cap,
                    device,
                )?;
                eprintln!(
                    "qwen3-asr: thinker backbone = {g:?} (quantized in {:.2?}, kv_cap={kv_cap})",
                    tq.elapsed()
                );
                m
            }
            None => {
                eprintln!("qwen3-asr: thinker backbone = bf16");
                Qwen3ASRModel::new(vb, &cfg, generation_config.eos_token_id.clone())?
            }
        };
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
            do_sample: generation_config.do_sample,
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
        mut on_partial: Option<&mut dyn FnMut(&str)>,
    ) -> Result<(String, usize, usize, usize, std::time::Duration)> {
        let seed = 34562u64;
        let prof = std::env::var("ASR_PROF").is_ok();
        // Honor do_sample: Qwen3-ASR ships do_sample=false (greedy). Passing the
        // config's temperature=1e-6 anyway would build Sampling::All, which does a
        // full-vocab divide+softmax, a ~607KB GPU→CPU readback and an O(V) CPU
        // multinomial per token. temperature=None → candle Sampling::ArgMax (one
        // argmax kernel + a 4-byte readback), which is what greedy decode means.
        let temperature = if self.do_sample { Some(self.temperature) } else { None };
        let mut logit_processor = get_logit_processor(temperature, None, None, seed);
        let i_start = Instant::now();
        // Opt ASR decode into the qwen3-tts fused SDPA path (simd_sum + fast::exp,
        // skips repeat_kv and its contiguous copy — both O(KV) per token). Prefill
        // (seq_len>1) auto-falls-back to eager inside eager_attention_forward's
        // is_decode check, so the guard is safe across the whole loop. The fused
        // kernel is not bit-identical to eager (why ASR was excluded by default),
        // but ASR is greedy (do_sample=false) so the numeric drift is irrelevant.
        let _sdpa_fast = sdpa_fast_guard();
        let mut generate: Vec<u32> = Vec::new();
        let mut prompt_tokens = 0usize;
        let mut fwd_time = std::time::Duration::ZERO;
        let mut smp_time = std::time::Duration::ZERO;
        // Per-token timing: prefill (first token = audio encoder + prompt prefill) vs
        // decode (each later token). Total-per-token (fwd + sample + any sync wait) is
        // the honest cost — the fwd/sample split is muddied by Metal async kernel launch.
        let mut prefill_tok = std::time::Duration::ZERO;
        let mut rest_tok = std::time::Duration::ZERO;
        let mut n_rest = 0usize;
        let mut is_first = true;
        for data in audio_datas.iter() {
            let mut input_ids = data.input_ids.clone();
            let mut input_features = Some(data.input_features.clone().to_dtype(self.dtype)?);
            let mut seq_len = input_ids.dim(1)?;
            prompt_tokens += seq_len;
            let mut seqlen_offset = 0;
            for _ in 0..max_tokens {
                let t_tok = Instant::now();
                let t_fwd = Instant::now();
                let logits =
                    self.model
                        .forward(&input_ids, seqlen_offset, input_features.as_ref())?;
                fwd_time += t_fwd.elapsed();
                let t_smp = Instant::now();
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
                smp_time += t_smp.elapsed();
                let tok_dt = t_tok.elapsed();
                if is_first {
                    prefill_tok = tok_dt;
                    is_first = false;
                } else {
                    rest_tok += tok_dt;
                    n_rest += 1;
                }
                generate.push(next_token);
                // Incremental transcript (Realtime input_audio_transcription.delta):
                // every couple of tokens decode the prefix so far and hand the cleaned
                // FULL transcript to the callback — the caller diffs it against the last
                // delta it emitted. Cheap (BPE detokenize of a short prefix), and decoding
                // the whole prefix (not just the new token) keeps multibyte boundaries sane.
                if let Some(cb) = on_partial.as_deref_mut() {
                    let emit = generate.len() % 2 == 0
                        || next_token == self.eos_token_id1
                        || next_token == self.eos_token_id2;
                    if emit
                        && let Ok(t) = self.tokenizer.token_decode(generate.clone())
                    {
                        let t = clean_asr_response(&t);
                        if !t.is_empty() {
                            cb(&t);
                        }
                    }
                }
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
        if prof {
            let rest_avg_ms = if n_rest > 0 {
                rest_tok.as_secs_f64() / n_rest as f64 * 1000.0
            } else {
                0.0
            };
            eprintln!(
                "asr prof: {} tokens | prefill {:?} | decode {}×{:.1}ms | fwd {:?} | sample {:?} | total {:?}",
                generate.len(), prefill_tok, n_rest, rest_avg_ms, fwd_time, smp_time, elapsed
            );
        }
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
            self.decode_audio_datas(&audio_datas, max_tokens, None)?;
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
        let (text, _, _, _, _) = self.decode_audio_datas(&[data], max_tokens, None)?;
        Ok(text)
    }

    /// Streaming variant of `transcribe_samples`: fires `on_partial` with the cleaned
    /// transcript-so-far every couple of generated tokens (the caller emits the suffix
    /// as Realtime `input_audio_transcription.delta`). The final transcript is returned
    /// as usual; `transcribe_samples` is this with `on_partial = None`.
    pub fn transcribe_samples_streaming(
        &mut self,
        samples_16k_mono: &[f32],
        max_tokens: usize,
        on_partial: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let audio = Tensor::new(samples_16k_mono, &self.device)?;
        let data =
            self.processor
                .process_audio_tensor(DEFAULT_TEMPLATE, &audio, &self.tokenizer)?;
        let (text, _, _, _, _) = self.decode_audio_datas(&[data], max_tokens, Some(on_partial))?;
        Ok(text)
    }
}

pub fn run(args: &[String]) -> Result<()> {
    // Pull `--quant <name>` / `--quant=<name>` out of the args; the rest are
    // positionals. Without `--quant`, fall back to TINY_CPM_ASR_QUANT via load().
    let mut quant: Option<GgmlDType> = None;
    let mut positional: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--quant" {
            let v = args
                .get(i + 1)
                .ok_or_else(|| anyhow!("--quant needs a value (q8_0 | q4_k | none)"))?;
            quant = parse_asr_quant(v)?;
            i += 2;
        } else if let Some(v) = a.strip_prefix("--quant=") {
            quant = parse_asr_quant(v)?;
            i += 1;
        } else {
            positional.push(a.clone());
            i += 1;
        }
    }
    const USAGE: &str =
        "usage: tiny-cpm asr qwen3 <model-dir> <audio-file> [max_tokens] [--quant q8_0|q4_k|none]";
    let Some(model_dir) = positional.first() else {
        bail!("{USAGE}");
    };
    let Some(audio_file) = positional.get(1) else {
        bail!("{USAGE}");
    };
    let max_tokens: usize = match positional.get(2) {
        Some(s) => s
            .parse()
            .map_err(|_| anyhow!("invalid max_tokens: {s}"))?,
        None => 512, // same default as aha's asr_audio
    };

    let device = Device::new_metal(0)?;

    let i_start = Instant::now();
    let mut engine = if let Some(q) = quant {
        Qwen3AsrEngine::load_with_quant(model_dir, &device, Some(q))?
    } else {
        Qwen3AsrEngine::load(model_dir, &device)? // honors TINY_CPM_ASR_QUANT
    };
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
