//! MiniCPM5-1B chat: accepts a pre-quantized `.gguf` (Q8_0) or a bf16
//! safetensors directory (auto-quantized to Q8_0 in-memory). Streams tokens to
//! stdout as they generate.
//!
//! Uses a vendored, patched `quantized_minicpm5` module — `quantized_llama` with
//! two fixes so MiniCPM5's non-standard head_dim (128 != hidden/heads = 96) works:
//! head_dim read from the GGUF, attention output reshaped to n_head*head_dim.
//!
//! Usage:
//!     tiny-cpm chat <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens]

use crate::quantized_minicpm5::ModelWeights;
use crate::token_output_stream::TokenOutputStream;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::io::Write;
use tokenizers::Tokenizer;

/// MiniCPM5 end-of-sequence token ids (from config.json: [1, 130073]).
const EOS_TOKEN_IDS: [u32; 2] = [1, 130073];

/// Result of one `generate_reply` call: clean text plus decode stats.
pub struct ChatGenStats {
    /// Full reply with `<think>`/`</think>` tags stripped.
    pub text: String,
    /// Tokens sampled in the decode loop (excludes the first prefill token).
    pub tokens: usize,
    /// Decode-loop wall time (tok/s = tokens / decode).
    pub decode: std::time::Duration,
}

/// Load MiniCPM5 from a `.gguf` file or a bf16 safetensors directory
/// (auto-quantized to Q8_0 in-memory) onto `device`.
pub fn load_model(model_path: &str, device: &Device) -> Result<ModelWeights> {
    let model = if model_path.ends_with(".gguf") {
        let mut file = std::fs::File::open(model_path)?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("gguf: {e:?}"))?;
        ModelWeights::from_gguf(content, &mut file, device)?
    } else {
        eprintln!("quantizing bf16 safetensors in {model_path} to Q8_0 in-memory...");
        ModelWeights::from_safetensors_dir(
            model_path,
            candle_core::quantized::GgmlDType::Q8_0,
            device,
        )?
    };
    Ok(model)
}

/// Generate one chat reply for `prompt` (ChatML single-turn, optional system
/// message). Clean streamed deltas (think tags stripped) are passed to `sink`;
/// the full clean text plus decode stats are returned. All diagnostics go to
/// stderr.
pub fn generate_reply(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    max_tokens: usize,
    sink: &mut dyn FnMut(&str),
) -> Result<ChatGenStats> {
    generate_reply_with_system(
        model, tokenizer, device, None, prompt, false, max_tokens, sink,
    )
}

/// `generate_reply` with an optional ChatML system message prepended.
/// `no_think` appends an empty think block (`<think>\n\n</think>\n\n`) after
/// the generation prompt — the official chat template's
/// `enable_thinking=false` convention, which makes MiniCPM5 skip reasoning
/// and answer directly (its untagged inner monologue otherwise leaks into the
/// reply).
pub fn generate_reply_with_system(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    device: &Device,
    system: Option<&str>,
    prompt: &str,
    no_think: bool,
    max_tokens: usize,
    sink: &mut dyn FnMut(&str),
) -> Result<ChatGenStats> {
    // MiniCPM5 uses ChatML. Render [optional system +] one user message +
    // generation prompt.
    let chat = match system {
        Some(sys) => format!(
            "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
        None => format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
    };
    let chat = if no_think {
        format!("{chat}<think>\n\n</think>\n\n")
    } else {
        chat
    };
    let enc = tokenizer
        .encode(chat, true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let prompt_tokens = enc.get_ids().to_vec();
    let n_prompt = prompt_tokens.len();
    eprintln!("prompt tokens: {n_prompt}");

    let mut lp = LogitsProcessor::from_sampling(
        299792458,
        Sampling::TopP {
            p: 0.9,
            temperature: 0.7,
        },
    );

    let mut tos = TokenOutputStream::new(tokenizer.clone());
    // Buffer to strip <think>/</think> tags from the stream.
    let mut full = String::new();
    let mut printed = 0usize;

    // Prefill the whole prompt, then sample the first token (streamed).
    let t0 = std::time::Instant::now();
    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?.squeeze(0)?;
    let mut next = lp.sample(&logits)?;
    eprintln!(
        "TTFT (prefill {n_prompt} tokens + first token): {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    let mut tokens = prompt_tokens.clone();
    tokens.push(next);
    if let Some(s) = tos.next_token(next)? {
        stream_clean(&mut full, &mut printed, &s, sink);
    }

    // Decode loop: stream each token (tags stripped) until EOS or max_tokens.
    let t_dec = std::time::Instant::now();
    let mut generated = 0usize;
    for index in 0..max_tokens {
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, n_prompt + index)?.squeeze(0)?;
        next = lp.sample(&logits)?;
        tokens.push(next);
        generated += 1;
        if let Some(s) = tos.next_token(next)? {
            stream_clean(&mut full, &mut printed, &s, sink);
        }
        if EOS_TOKEN_IDS.contains(&next) {
            break;
        }
    }
    if let Some(rest) = tos.decode_rest()? {
        stream_clean(&mut full, &mut printed, &rest, sink);
    }
    let dt = t_dec.elapsed().as_secs_f64();
    eprintln!(
        "\ndecode: {generated} tokens in {:.2}s ({:.1} tok/s)",
        dt,
        generated as f64 / dt
    );
    let text = full.replace("<think>", "").replace("</think>", "");
    Ok(ChatGenStats {
        text,
        tokens: generated,
        decode: t_dec.elapsed(),
    })
}

/// Append `s` to `full`, strip `<think>`/`</think>` tags, and pass any newly
/// available clean text to `sink`. `printed` tracks bytes already emitted
/// (always at a UTF-8 char boundary, since clean grows only by complete
/// decoded strings).
fn stream_clean(full: &mut String, printed: &mut usize, s: &str, sink: &mut dyn FnMut(&str)) {
    full.push_str(s);
    let clean = full.replace("<think>", "").replace("</think>", "");
    if clean.len() > *printed {
        sink(&clean[*printed..]);
        *printed = clean.len();
    }
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        eprintln!(
            "usage: tiny-cpm chat <model.gguf | bf16-dir> <tokenizer.json> <prompt> [max_tokens]"
        );
        std::process::exit(1);
    }
    let model_path = &args[0];
    let tok_path = &args[1];
    let prompt = &args[2];
    let max_tokens: usize = args.get(3).map(|s| s.parse().unwrap_or(512)).unwrap_or(512);

    let t_load = std::time::Instant::now();
    let device = Device::new_metal(0)?;
    let mut model = load_model(model_path, &device)?;
    eprintln!("loaded model in {:.2}s", t_load.elapsed().as_secs_f64());

    let tokenizer =
        Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let mut sink = |s: &str| {
        print!("{s}");
        let _ = std::io::stdout().flush();
    };
    let _stats = generate_reply(
        &mut model, &tokenizer, &device, prompt, max_tokens, &mut sink,
    )?;
    std::io::stdout().flush()?;
    Ok(())
}
