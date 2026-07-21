//! tiny-llama: MiniCPM5-1B inference on Metal via the official `candle` crate.
//! Accepts a pre-quantized `.gguf` (Q8_0) or a bf16 safetensors directory
//! (auto-quantized to Q8_0 in-memory). Streams tokens to stdout as they generate.
//!
//! Uses a vendored, patched `quantized_minicpm5` module — `quantized_llama` with
//! two fixes so MiniCPM5's non-standard head_dim (128 != hidden/heads = 96) works:
//! head_dim read from the GGUF, attention output reshaped to n_head*head_dim.
//!
//! Usage:
//!     cargo run --release -- <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens]

mod quantized_minicpm5;
mod token_output_stream;

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <model.gguf | bf16-dir> <tokenizer.json> <prompt> [max_tokens]",
            args[0]
        );
        std::process::exit(1);
    }
    let model_path = &args[1];
    let tok_path = &args[2];
    let prompt = &args[3];
    let max_tokens: usize = args
        .get(4)
        .map(|s| s.parse().unwrap_or(512))
        .unwrap_or(512);

    let t_load = std::time::Instant::now();
    let device = Device::new_metal(0)?;
    let mut model = if model_path.ends_with(".gguf") {
        let mut file = std::fs::File::open(model_path)?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("gguf: {e:?}"))?;
        ModelWeights::from_gguf(content, &mut file, &device)?
    } else {
        eprintln!("quantizing bf16 safetensors in {model_path} to Q8_0 in-memory...");
        ModelWeights::from_safetensors_dir(
            model_path,
            candle_core::quantized::GgmlDType::Q8_0,
            &device,
        )?
    };
    eprintln!("loaded model in {:.2}s", t_load.elapsed().as_secs_f64());

    let tokenizer =
        Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // MiniCPM5 uses ChatML. Render a single-turn user message + generation prompt.
    let chat = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
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

    let mut tos = TokenOutputStream::new(tokenizer);
    // Buffer to strip <think>/</think> tags from the stream.
    let mut full = String::new();
    let mut printed = 0usize;

    // Prefill the whole prompt, then sample the first token (streamed).
    let t0 = std::time::Instant::now();
    let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?.squeeze(0)?;
    let mut next = lp.sample(&logits)?;
    eprintln!(
        "TTFT (prefill {n_prompt} tokens + first token): {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    let mut tokens = prompt_tokens.clone();
    tokens.push(next);
    if let Some(s) = tos.next_token(next)? {
        stream_clean(&mut full, &mut printed, &s)?;
    }

    // Decode loop: stream each token (tags stripped) until EOS or max_tokens.
    let t_dec = std::time::Instant::now();
    let mut generated = 0usize;
    for index in 0..max_tokens {
        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, n_prompt + index)?.squeeze(0)?;
        next = lp.sample(&logits)?;
        tokens.push(next);
        generated += 1;
        if let Some(s) = tos.next_token(next)? {
            stream_clean(&mut full, &mut printed, &s)?;
        }
        if EOS_TOKEN_IDS.contains(&next) {
            break;
        }
    }
    if let Some(rest) = tos.decode_rest()? {
        stream_clean(&mut full, &mut printed, &rest)?;
    }
    std::io::stdout().flush()?;
    let dt = t_dec.elapsed().as_secs_f64();
    eprintln!(
        "\ndecode: {generated} tokens in {:.2}s ({:.1} tok/s)",
        dt,
        generated as f64 / dt
    );
    Ok(())
}

/// Append `s` to `full`, strip `<think>`/`</think>` tags, and print any newly
/// available clean text. `printed` tracks bytes already emitted (always at a
/// UTF-8 char boundary, since clean grows only by complete decoded strings).
fn stream_clean(full: &mut String, printed: &mut usize, s: &str) -> Result<()> {
    full.push_str(s);
    let clean = full.replace("<think>", "").replace("</think>", "");
    if clean.len() > *printed {
        print!("{}", &clean[*printed..]);
        std::io::stdout().flush()?;
        *printed = clean.len();
    }
    Ok(())
}
