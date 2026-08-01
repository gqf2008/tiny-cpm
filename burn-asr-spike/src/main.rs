//! burn spike CLI: `burn-asr-spike <model-dir> <audio-file> [max_tokens]`
//!
//! Ports tiny-cpm's `asr qwen3` driver to burn 0.22.0-pre.1 on Metal:
//! safetensors (bf16 → f16) → mel frontend (CPU) → greedy decode (per-step
//! GPU forward, logits read back like candle's LogitsProcessor) → transcript
//! on stdout, timings on stderr.
//!
//! Benchmark protocol: run 1 = warmup (MSL shader compile + matmul autotune),
//! run 2 = measured. Both runs share the same audio.

#![recursion_limit = "256"]

mod audio;
mod config;
mod model;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Result, bail};
use burn::tensor::{Device, DeviceKind, DType, Int, Tensor, TensorData};
use model::Weights;
use tokenizers::Tokenizer;

/// aha generate.rs default_template, same as tiny-cpm's exec/qwen3_asr.rs.
const DEFAULT_TEMPLATE: &str = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n";

/// aha clean_asr_response: text after "<asr_text>" (or trimmed raw).
fn clean_asr_response(raw: &str) -> String {
    if let Some(start) = raw.find("<asr_text>") {
        raw[start + "<asr_text>".len()..].trim().to_string()
    } else {
        raw.trim().to_string()
    }
}

/// replace_special_tokens: expand <|audio_pad|> to token_len copies.
fn replace_special_tokens(text: &str, token_len: usize) -> String {
    let replace = "<|audio_placeholder|>".repeat(token_len);
    let text = text.replacen("<|audio_pad|>", &replace, 1);
    text.replace("<|audio_placeholder|>", "<|audio_pad|>")
}

/// Load all *.safetensors in a dir, converting every tensor to f16.
/// (candle mmaps bf16; burn's Metal backend has no BF16, and f16 is the native
/// Apple GPU type, so we convert once at load. Memory footprint identical.)
fn load_weights(model_dir: &str) -> Result<HashMap<String, TensorData>> {
    let mut files: Vec<String> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().is_file()
                && e.path()
                    .extension()
                    .is_some_and(|x| x == "safetensors")
        })
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no safetensors files found in {model_dir}");
    }
    let mut out = HashMap::new();
    for f in &files {
        let bytes = std::fs::read(f)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("safetensors {f}: {e}"))?;
        for (name, t) in st.tensors() {
            let dt = match t.dtype() {
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F64 => DType::F64,
                safetensors::Dtype::I64 => DType::I64,
                safetensors::Dtype::I32 => DType::I32,
                other => bail!("unsupported safetensors dtype {other:?} in {f}"),
            };
            let target = if std::env::var("BURN_ASR_F32").is_ok() {
                DType::F32
            } else {
                DType::F16
            };
            let td = TensorData::from_bytes_vec(t.data().to_vec(), t.shape(), dt)
                .convert_dtype(target);
            out.insert(name, td);
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(model_dir) = args.get(1) else {
        bail!("usage: burn-asr-spike <model-dir> <audio-file> [max_tokens]");
    };
    let Some(audio_file) = args.get(2) else {
        bail!("usage: burn-asr-spike <model-dir> <audio-file> [max_tokens]");
    };
    let max_tokens: usize = match args.get(3) {
        Some(s) => s.parse().map_err(|_| anyhow::anyhow!("invalid max_tokens: {s}"))?,
        None => 512,
    };

    let device = Device::metal(DeviceKind::DefaultDevice);
    let mdtype = if std::env::var("BURN_ASR_F32").is_ok() { "f32" } else { "f16" };
    eprintln!("backend: burn {} (wgpu/cubecl, Metal), model dtype {mdtype}", env!("CARGO_PKG_VERSION"));
    eprintln!("device: {device:?}");

    // configs
    let cfg: config::Qwen3ASRConfig =
        serde_json::from_slice(&std::fs::read(format!("{model_dir}/config.json"))?)?;
    let gen_cfg: config::GenerationConfig =
        serde_json::from_slice(&std::fs::read(format!("{model_dir}/generation_config.json"))?)?;
    let fe: config::FeatureExtractorConfig = serde_json::from_slice(&std::fs::read(format!(
        "{model_dir}/preprocessor_config.json"
    ))?)?;
    if gen_cfg.eos_token_id.len() < 2 {
        bail!("generation_config.json: expected 2 eos_token_id entries");
    }
    let tokenizer = load_tokenizer(model_dir)?;

    // weights + model
    let t0 = Instant::now();
    let weights = load_weights(model_dir)?;
    let n_tensors = weights.len();
    let w = Weights::new(weights, device.clone());
    let mut model = model::build_model(&w, &cfg.thinker_config)?;
    eprintln!(
        "loaded model in {:?} ({} tensors, {} safetensors shard(s), bf16->f16)",
        t0.elapsed(),
        n_tensors,
        1
    );

    // audio → mel → prompt
    let t1 = Instant::now();
    let (mut samples, sr) = audio::decode_audio(audio_file)?;
    if sr != audio::TARGET_SR {
        samples = audio::resample_sinc(&samples, sr, audio::TARGET_SR);
    }
    let audio_secs = samples.len() as f64 / audio::TARGET_SR as f64;
    let chunks: Vec<Vec<f32>> = audio::split_into_chunks(&samples);
    let mut datas = Vec::new();
    for chunk in &chunks {
        let (mel, n_frames) = audio::compute_mel(chunk)?;
        let output_len = audio::get_feat_extract_output_lengths(n_frames);
        let text = replace_special_tokens(DEFAULT_TEMPLATE, output_len);
        let ids = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec();
        let mel_td = TensorData::new(mel, [128, n_frames]);
        let mel_t = Tensor::from_data(mel_td, (&device, if std::env::var("BURN_ASR_F32").is_ok() { DType::F32 } else { DType::F16 }));
        datas.push((ids, mel_t));
    }
    let mel_ms = t1.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "audio: {:.1}s @ {}Hz, {} chunk(s), mel in {:.1}ms, prompt tokens: {}",
        audio_secs,
        sr,
        chunks.len(),
        mel_ms,
        datas[0].0.len()
    );

    // warmup run (MSL compile + autotune), then measured run
    let warm = transcribe(&mut model, &datas, max_tokens, &gen_cfg.eos_token_id, &device, &tokenizer)?;
    eprintln!(
        "[warmup] {} chunk(s), {} generated tokens in {:?} ({:.2} tok/s)",
        warm.chunks,
        warm.tokens,
        warm.elapsed,
        warm.tokens as f64 / warm.elapsed.as_secs_f64()
    );

    let run = transcribe(&mut model, &datas, max_tokens, &gen_cfg.eos_token_id, &device, &tokenizer)?;
    let tok_s = run.tokens as f64 / run.elapsed.as_secs_f64();
    let rtf = run.elapsed.as_secs_f64() / audio_secs;
    eprintln!(
        "decode: {} chunk(s), {} prompt tokens, {} generated tokens in {:?} ({:.2} tok/s, RTF {:.3})",
        run.chunks,
        run.prompt_tokens,
        run.tokens,
        run.elapsed,
        tok_s,
        rtf
    );
    println!("{}", run.text);
    Ok(())
}

struct RunStats {
    chunks: usize,
    prompt_tokens: usize,
    tokens: usize,
    text: String,
    elapsed: std::time::Duration,
}

fn transcribe(
    model: &mut model::Qwen3AsrModel,
    datas: &[(Vec<u32>, Tensor<2>)],
    max_tokens: usize,
    eos: &[u32],
    device: &Device,
    tokenizer: &Tokenizer,
) -> Result<RunStats> {
    let t0 = Instant::now();
    let mut generate: Vec<u32> = Vec::new();
    let mut prompt_tokens = 0usize;
    for (ids, features) in datas {
        let mut input_ids = Tensor::<1, Int>::from_ints(ids.as_slice(), device).reshape([1, ids.len()]); // (1, seq)
        let mut input_features = Some(features.clone());
        let mut seq_len = ids.len();
        prompt_tokens += seq_len;
        let mut seqlen_offset = 0usize;
        for _ in 0..max_tokens {
            let logits = model.forward(input_ids.clone(), seqlen_offset, input_features.take())?;
            let logits = logits
                .cast(DType::F32)
                .to_data()
                .to_vec::<f32>()?;
            let next = argmax(&logits);
            generate.push(next);
            if eos.contains(&next) {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_ints([[next as i32]], device);
        }
        model.clear_cache();
    }
    // force all pending GPU work to finish before the clock stops
    device
        .sync()
        .map_err(|e| anyhow::anyhow!("device sync: {e}"))?;
    let elapsed = t0.elapsed();
    let raw = tokenizer
        .decode(&generate, true)
        .map_err(|e| anyhow::anyhow!("token decode: {e}"))?;
    let text = clean_asr_response(&raw);
    Ok(RunStats {
        chunks: datas.len(),
        prompt_tokens,
        tokens: generate.len(),
        text,
        elapsed,
    })
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// tokenizer.json if present, else BPE from vocab.json + merges.txt with the
/// added-tokens table from tokenizer_config.json (tiny-cpm TokenizerModel::init).
fn load_tokenizer(model_dir: &str) -> Result<Tokenizer> {
    let tokenizer_file = format!("{model_dir}/tokenizer.json");
    if std::path::Path::new(&tokenizer_file).exists() {
        return Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("tokenizer from file: {e}"));
    }
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::AddedToken;

    let vocab_file = format!("{model_dir}/vocab.json");
    let merges_file = format!("{model_dir}/merges.txt");
    if !std::path::Path::new(&vocab_file).exists() {
        bail!("neither tokenizer.json nor vocab.json found in {model_dir}");
    }
    let bpe = BPE::from_file(&vocab_file, &merges_file)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build BPE tokenizer: {e}"))?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    let config_file = format!("{model_dir}/tokenizer_config.json");
    if std::path::Path::new(&config_file).exists() {
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        if let Some(added_tokens_decoder) = config.get("added_tokens_decoder") {
            let mut special_tokens = Vec::new();
            if let serde_json::Value::Object(tokens_map) = added_tokens_decoder {
                for (_, token_info) in tokens_map {
                    if let serde_json::Value::Object(token_obj) = token_info
                        && let Some(content) = token_obj.get("content").and_then(|v| v.as_str())
                    {
                        let special = token_obj
                            .get("special")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        special_tokens.push(AddedToken::from(content.to_string(), special));
                    }
                }
            }
            tokenizer.add_tokens(&special_tokens);
        }
    }
    Ok(tokenizer)
}
