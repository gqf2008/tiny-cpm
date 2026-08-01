//! MiniCPM5-1B chat: accepts a pre-quantized `.gguf` (Q8_0) or a bf16
//! safetensors directory (auto-quantized in-memory, default Q8_0; override
//! with `TINY_CPM_QUANT` or `--quant <name>`). Streams tokens to stdout as
//! they generate.
//!
//! Uses a vendored, patched `quantized_minicpm5` module — `quantized_llama`
//! with fixes for MiniCPM5's non-standard head_dim (128 != hidden/heads = 96)
//! and its NEOX (non-interleaved) RoPE convention.
//!
//! Usage:
//!     tiny-cpm chat <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens] [--quant <name>]

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
/// (auto-quantized in-memory) onto `device`. Quant level comes from
/// `TINY_CPM_QUANT` (default Q8_0). See [`load_model_with_quant`].
pub fn load_model(model_path: &str, device: &Device) -> Result<ModelWeights> {
    load_model_with_quant(model_path, None, device)
}

/// `load_model` with an explicit quant level (`quant_name`: e.g. "q4_k_m",
/// "q8_0"). Priority: `quant_name` > `TINY_CPM_QUANT` env > Q8_0. A `.gguf`
/// path ignores the quant level (the file is already quantized).
pub fn load_model_with_quant(
    model_path: &str,
    quant_name: Option<&str>,
    device: &Device,
) -> Result<ModelWeights> {
    let model = if model_path.ends_with(".gguf") {
        let mut file = std::fs::File::open(model_path)?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| anyhow::anyhow!("gguf: {e:?}"))?;
        ModelWeights::from_gguf(content, &mut file, device)?
    } else {
        // Runtime quantization of the bf16 safetensors. Default Q8_0; override
        // via `quant_name` or TINY_CPM_QUANT (q8_0, q4_k_m, q5_k_m, q6_k, ...).
        // K-quants map to candle's `GgmlDType::QnK` block formats.
        let dtype = match quant_name {
            Some(q) => parse_quant_source(Some(q), "--quant"),
            None => parse_quant_source(
                std::env::var("TINY_CPM_QUANT").ok().as_deref(),
                "TINY_CPM_QUANT",
            ),
        };
        eprintln!("quantizing bf16 safetensors in {model_path} to {dtype:?} in-memory...");
        ModelWeights::from_safetensors_dir(model_path, dtype, device)?
    };
    Ok(model)
}

/// Map a quant name to a `GgmlDType` (default Q8_0), reporting the value as
/// coming from `--quant`. See [`parse_quant_source`].
pub fn parse_quant(name: Option<&str>) -> candle_core::quantized::GgmlDType {
    parse_quant_source(name, "--quant")
}

/// `parse_quant` with a `source` label for the unknown-name warning. Accepted
/// names: q8_0/q4_0/q4_1/q5_0/q5_1/q4_k/q5_k/q6_k/q3_k/q2_k/f16/f32 (plus
/// aliases such as `q8`, `q4_k_m`, `q4k`).
fn parse_quant_source(name: Option<&str>, source: &str) -> candle_core::quantized::GgmlDType {
    use candle_core::quantized::GgmlDType::*;
    match name.map(|s| s.trim().to_ascii_lowercase()) {
        None => Q8_0,
        Some(s) => match s.as_str() {
            "q8_0" | "q8" => Q8_0,
            "q4_0" => Q4_0,
            "q4_1" => Q4_1,
            "q5_0" => Q5_0,
            "q5_1" => Q5_1,
            "q4_k" | "q4_k_m" | "q4k" => Q4K,
            "q5_k" | "q5_k_m" | "q5k" => Q5K,
            "q6_k" | "q6k" => Q6K,
            "q3_k" | "q3_k_m" | "q3k" => Q3K,
            "q2_k" | "q2k" => Q2K,
            "f16" => F16,
            "f32" => F32,
            other => {
                eprintln!("unknown quant '{other}' (from {source}), defaulting to Q8_0");
                Q8_0
            }
        },
    }
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
        model, tokenizer, device, None, prompt, false, max_tokens, sink, None,
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
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<ChatGenStats> {
    generate_reply_with_history(
        model,
        tokenizer,
        device,
        system,
        &[],
        prompt,
        no_think,
        max_tokens,
        sink,
        cancel,
    )
}

/// Multi-turn ChatML: renders [optional system +] prior `history` turns
/// (alternating user/assistant) + the current `prompt` + the generation
/// prompt. `history` items are `(is_user, text)`. The server is stateless —
/// the caller (browser) owns the history and passes it per turn.
pub fn generate_reply_with_history(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    device: &Device,
    system: Option<&str>,
    history: &[(bool, String)],
    prompt: &str,
    no_think: bool,
    max_tokens: usize,
    sink: &mut dyn FnMut(&str),
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<ChatGenStats> {
    let mut chat = String::new();
    if let Some(sys) = system {
        chat.push_str("<|im_start|>system\n");
        chat.push_str(sys);
        chat.push_str("<|im_end|>\n");
    }
    for (is_user, text) in history {
        chat.push_str("<|im_start|>");
        chat.push_str(if *is_user { "user" } else { "assistant" });
        chat.push('\n');
        chat.push_str(text);
        chat.push_str("<|im_end|>\n");
    }
    chat.push_str("<|im_start|>user\n");
    chat.push_str(prompt);
    chat.push_str("<|im_end|>\n<|im_start|>assistant\n");
    let chat = if no_think {
        format!("{chat}<think>\n\n</think>\n\n")
    } else {
        format!("{chat}<think>\n")
    };
    eprintln!(
        "=== ChatML ({} chars) ===\n{chat}\n=== end ChatML ===",
        chat.len()
    );
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
    // Repetition-guard kill-switch: TINY_CPM_REPEAT_GUARD=0/off/false/no disables it
    // (same value convention as the qwen3-tts env knobs).
    let repeat_guard = match std::env::var("TINY_CPM_REPEAT_GUARD") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "off" | "false" | "no"
        ),
        Err(_) => true,
    };
    for index in 0..max_tokens {
        if let Some(c) = cancel
            && c.load(std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("llm: cancelled by barge-in after {generated} tokens");
            break;
        }
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
        // Repetition guard: small models (1B) sometimes fall into an exact
        // repeat loop ("星期五，星期五，…") on questions they can't answer and
        // would otherwise spew it until max_tokens — and the TTS then reads the
        // whole loop aloud. Break as soon as a short trailing phrase repeats
        // back-to-back several times. Only the visible reply text is checked:
        // in think mode that is everything after `</think>`, so a loop inside
        // the reasoning block cannot kill the reply before the answer appears.
        // (In `no_think` mode the think block is pre-closed in the prompt, so
        // the whole generated text is the reply.)
        let reply = if no_think { &full } else { after_think(&full) };
        if generated % 8 == 0 && repeat_guard && is_repeating(reply) {
            eprintln!("llm: repetition loop detected, stopping at {generated} tokens");
            break;
        }
    }
    if let Some(rest) = tos.decode_rest()? {
        stream_clean(&mut full, &mut printed, &rest, sink);
    }
    // Fallback: if the think block never closed (model hit max_tokens or EOS
    // mid-think), emit the accumulated text stripped of tags so the user
    // still gets SOMETHING.
    if printed == 0 && !full.is_empty() {
        let fallback = full.replace("<think>", "").replace("</think>", "");
        if !fallback.is_empty() {
            sink(&fallback);
            printed = fallback.len();
        }
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

/// The visible reply text: everything after the first `</think>` — the same
/// text `stream_clean` emits. Before the think block closes there is no reply
/// yet, so this is empty.
fn after_think(full: &str) -> &str {
    match full.find("</think>") {
        Some(idx) => &full[idx + 8..], // 8 = len("</think>")
        None => "",
    }
}

/// Append `s` to `full`, strip `<think>`/`</think>` tags, and pass any newly
/// available clean text to `sink`. `printed` tracks bytes already emitted
/// (always at a UTF-8 char boundary, since clean grows only by complete
/// decoded strings).
fn stream_clean(full: &mut String, printed: &mut usize, s: &str, sink: &mut dyn FnMut(&str)) {
    full.push_str(s);
    // Suppress think block: emit only text AFTER </think> (the actual response).
    // Before </think> is reasoning content — never displayed or spoken.
    let clean = after_think(full);
    if clean.len() > *printed {
        sink(&clean[*printed..]);
        *printed = clean.len();
    }
}

/// Detect an exact back-to-back repeat loop at the tail of `text`. Tries short
/// phrase lengths (in chars) and returns true if the same phrase repeats ≥4
/// times consecutively at the end. Cheap: only inspects the tail (at most the
/// last 48 chars — the longest window that can hold 4 repeats of the longest
/// tested phrase length, 12 × 4).
fn is_repeating(text: &str) -> bool {
    // Collect only the tail instead of the whole text: `rev().take(48)` is
    // O(48) per call (the guard runs every 8 tokens during decode).
    let mut chars: Vec<char> = text.chars().rev().take(48).collect();
    chars.reverse();
    let n = chars.len();
    if n < 16 {
        return false;
    }
    for plen in [4usize, 6, 8, 12] {
        if n < plen * 4 {
            continue;
        }
        let tail = &chars[n - plen..];
        let mut count = 1;
        let mut i = n - plen;
        while i >= plen {
            let prev = &chars[i - plen..i];
            if prev == tail {
                count += 1;
                i -= plen;
            } else {
                break;
            }
        }
        if count >= 4 {
            return true;
        }
    }
    false
}

pub fn run(args: &[String]) -> Result<()> {
    // `--quant <name>` / `--quant=<name>` may appear anywhere in args;
    // everything else is a positional: <model.gguf | bf16-dir> <tokenizer.json>
    // <prompt> [max_tokens].
    let mut quant: Option<String> = None;
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--quant" {
            i += 1;
            let value = args
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("--quant requires a value, e.g. --quant q8_0"))?;
            quant = Some(value.clone());
        } else if let Some(value) = args[i].strip_prefix("--quant=") {
            quant = Some(value.to_string());
        } else {
            positional.push(args[i].as_str());
        }
        i += 1;
    }
    if positional.len() < 3 {
        eprintln!(
            "usage: tiny-cpm chat <model.gguf | bf16-dir> <tokenizer.json> <prompt> [max_tokens] [--quant <name>]"
        );
        std::process::exit(1);
    }
    if positional.len() > 4 {
        return Err(anyhow::anyhow!(
            "too many positional arguments for chat (expected <model> <tokenizer> <prompt> [max_tokens], plus optional --quant)"
        ));
    }
    let model_path = positional[0];
    let tok_path = positional[1];
    let prompt = positional[2];
    // Strict max_tokens: a typo'd 4th argument must error instead of silently
    // becoming 512.
    let max_tokens = match positional.get(3) {
        Some(s) => s.parse::<usize>().map_err(|e| {
            anyhow::anyhow!("invalid max_tokens '{s}': {e} (expected a non-negative integer)")
        })?,
        None => 512,
    };
    if quant.is_some() && model_path.ends_with(".gguf") {
        eprintln!("warning: --quant is ignored for .gguf files (the file is already quantized)");
    }

    let t_load = std::time::Instant::now();
    let device = Device::new_metal(0)?;
    let mut model = load_model_with_quant(model_path, quant.as_deref(), &device)?;
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

#[cfg(test)]
mod tests {
    use super::{after_think, is_repeating};

    // ── is_repeating boundaries ─────────────────────────────────────────────

    /// Shorter than 16 chars → never flagged.
    #[test]
    fn too_short_text_is_not_repeating() {
        assert!(!is_repeating("abcabcabcabc")); // 12 chars
    }

    /// Boundary n == plen*4 at the smallest phrase length (4 × 4 = 16 chars).
    #[test]
    fn exactly_four_repeats_at_n_equals_plen_times_4() {
        assert!(is_repeating("abcdabcdabcdabcd"));
    }

    /// Exactly 3 back-to-back repeats (plen=8, 24 chars) stays below the ≥4
    /// threshold; no shorter phrase length aligns on this text.
    #[test]
    fn exactly_three_repeats_is_not_repeating() {
        assert!(!is_repeating("abcdefghabcdefghabcdefgh"));
    }

    /// Exactly 4 back-to-back repeats (plen=8, 32 chars = plen*4) → flagged.
    #[test]
    fn exactly_four_repeats_is_repeating() {
        assert!(is_repeating("abcdefghabcdefghabcdefghabcdefgh"));
    }

    /// Longest tested phrase length (plen=12): 3 repeats false, 4 true.
    #[test]
    fn twelve_char_phrase_boundary() {
        assert!(!is_repeating("abcdefghijklabcdefghijklabcdefghijkl")); // 36 chars
        assert!(is_repeating(
            "abcdefghijklabcdefghijklabcdefghijklabcdefghijkl"
        )); // 48 chars
    }

    /// Multibyte (UTF-8) chars are counted as chars, not bytes.
    #[test]
    fn multibyte_repeats() {
        // 4 repeats of a 6-char phrase (24 chars) → flagged.
        assert!(is_repeating(
            "我喜欢喝奶茶我喜欢喝奶茶我喜欢喝奶茶我喜欢喝奶茶"
        ));
        // Exactly 3 repeats of a 6-char phrase (18 chars) → not flagged.
        assert!(!is_repeating("今天天气很好今天天气很好今天天气很好"));
        // Long, non-repeating multibyte text → not flagged.
        assert!(!is_repeating(
            "这是一个很长的句子，用来确认多字节字符不会误触发重复检测逻辑。"
        ));
    }

    // ── after_think (post-think-only scope) ─────────────────────────────────

    /// Only text after the first `</think>` counts as the reply.
    #[test]
    fn after_think_returns_only_post_think_text() {
        assert_eq!(after_think("<think>思考过程</think>最终答案"), "最终答案");
        assert_eq!(after_think("还没有关闭的思考块"), "");
        assert_eq!(after_think(""), "");
    }

    /// A repeat loop inside `<think>` must not trip the guard: the call site
    /// feeds `after_think(&full)`, so the reasoning block is invisible to it.
    #[test]
    fn repetition_inside_think_block_is_ignored() {
        let full = format!("<think>{}</think>这是一个简短的回答。", "ab".repeat(40));
        assert!(!is_repeating(after_think(&full)));
    }

    /// ...while a repeat in the post-think reply is still caught.
    #[test]
    fn repetition_after_think_block_is_caught() {
        let full = format!("<think>推理</think>{}", "abcd".repeat(4));
        assert!(is_repeating(after_think(&full)));
    }
}
