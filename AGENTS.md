# AGENTS.md

Guidance for AI coding agents working on this repository. Assumes no prior knowledge of the project.

## Project overview

`tiny-cpm` is a single-binary Rust CLI that runs [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) — a reasoning ("thinking") LLM — **quantized (GGUF, e.g. Q8_0)** on Apple **Metal**, using the official [`candle`](https://github.com/huggingface/candle) crate (0.11, from crates.io, not a fork) plus a vendored, minimally patched `quantized_minicpm5` model module.

Key characteristics:

- **Single-shot, synchronous CLI**: load → prefill → decode → print → exit. No server, no batching, no streaming protocol — tokens are just printed to stdout as they generate.
- MiniCPM5's `<think>…</think>` reasoning tags are stripped inline during streaming, so the visible output reads as one continuous answer.
- ~57–59 tok/s decode on Metal with Q8_0 (vs ~24 tok/s for bf16). Candle's Metal quantized path dequantizes per block then matmuls; for a 1B model the memory-bandwidth win from half-size weights beats the dequant cost.
- Two load paths from the same binary, auto-detected from `argv[1]`:
  - a pre-converted `.gguf` file → `ModelWeights::from_gguf` (fast load);
  - a bf16 safetensors directory (HF layout: sharded `.safetensors` + `config.json`) → `ModelWeights::from_safetensors_dir`, which mmaps bf16 tensors on CPU and quantizes each to Q8_0 onto Metal at load time (~5 s slower, no llama.cpp needed). Both produce equivalent Q8 weights.

## Tech stack

- **Rust**, edition 2024 (see `Cargo.toml`; builds with stable rustc ≥ 1.85-ish — verified with 1.97).
- **Target platform: macOS / Apple Silicon only.** `candle-core` and `candle-nn` are built with the `metal` feature, and `main.rs` calls `Device::new_metal(0)` unconditionally.
- Dependencies (`Cargo.toml`): `candle-core`/`candle-nn`/`candle-transformers` 0.11, `anyhow`, `serde` + `serde_json` (config.json parsing), `tokenizers` 0.22, `tracing`.
- Model weights are **not** in the repo. You need MiniCPM5-1B weights plus `tokenizer.json`, either as a GGUF (converted with llama.cpp) or a bf16 safetensors directory.

## Prerequisites

macOS needs the Metal Toolchain (one-time):

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Build, run, and test commands

```bash
cargo run --release -- <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens]

# Examples (paths assume a local ./models/ directory, not committed):
cargo run --release -- ./models/MiniCPM5-1B-Q8_0.gguf ./models/tokenizer.json "What is AI?" 512
cargo run --release -- ./models/minicpm5-1b/ ./models/tokenizer.json "What is AI?" 512

cargo build --release   # build without running
cargo check             # fast type-check
cargo fmt
cargo clippy
cargo test              # runs the mask unit tests in src/quantized_minicpm5.rs
```

The CLI is positional (parsed in `main()` in `src/main.rs`):

| argv  | meaning                            | required               |
|-------|------------------------------------|------------------------|
| `[1]` | `.gguf` file **or** bf16 directory | yes                    |
| `[2]` | `tokenizer.json`                   | yes                    |
| `[3]` | prompt                             | yes                    |
| `[4]` | max tokens to generate             | no (default 512)       |

Output contract: generated text goes to **stdout**; all diagnostics (load time, prompt token count, TTFT, decode tok/s) go to **stderr**. Keep it that way — the stdout stream is the "clean answer" consumers may pipe.

## Code organization

Three source files, no submodules:

- **`src/main.rs`** — thin driver. Parses CLI args → picks the load path by file extension → renders MiniCPM5's ChatML template (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`) → prefill → top-p decode loop → streams tokens with reasoning tags stripped (`stream_clean`) → prints timing. All model math lives in the vendored module, not here.
- **`src/quantized_minicpm5.rs`** — the model. A vendored copy of `candle_transformers::models::quantized_llama` (0.11) with two patches (see below). Supports 2/3/4/8-bit GGUF (Q8_0, Q4_K_M, …), GQA, an in-layer KV cache, and a mask cache keyed by `(seq_len, kv_len)`. `MAX_SEQ_LEN = 4096`. Also contains the project's only tests.
- **`src/token_output_stream.rs`** — vendored copy of candle's streaming tokenizer wrapper; decodes tokens incrementally so partial words aren't flushed mid-token. Kept identical to upstream.

### The two MiniCPM5 patches (why the model is vendored)

Upstream `quantized_llama` assumes `head_dim == hidden_size / num_heads` and `num_heads * head_dim == hidden_size`. MiniCPM5 has `head_dim = 128`, `hidden_size = 1536`, `num_heads = 16`, so `num_heads * head_dim = 2048 ≠ 1536`. Both assumptions break. The vendored module:

1. Reads `head_dim` from GGUF metadata (`llama.rope.dimension_count`) instead of `embedding_length / head_count` — or from `config.json`'s `head_dim` field on the safetensors path.
2. Reshapes the attention output to `num_heads * head_dim` (2048) before the output projection, which maps 2048 → 1536 (`hidden`).

It also detects the RoPE convention from `general.architecture` on the GGUF path (NEOX for qwen/phi/falcon/…, NORM/interleaved otherwise); MiniCPM5 uses NORM.

## Testing strategy

- The only automated tests are 6 CPU-only unit tests at the bottom of `src/quantized_minicpm5.rs` (`#[cfg(test)] mod tests`). They verify causal-mask shapes, values, and broadcast compatibility (covering a past panic when `index_pos > 0`). Run with `cargo test`.
- There is no integration test, CI pipeline, or deployment process — this is a local single-binary tool. Verification beyond the unit tests is empirical: run the CLI against a real model and check output quality and tok/s.

## Code conventions

- **Keep `src/quantized_minicpm5.rs` a minimal diff from upstream `quantized_llama`** (only the two patches described above), so it is easy to re-vendor when candle releases updates. The same applies to `src/token_output_stream.rs` — do not restyle vendored code.
- **`src/main.rs` stays a thin driver**; don't move model math into it.
- Comments and docs are in English; match the existing terse, technical comment style (module-level `//!` docs explain *why*, not *what*).
- Error handling: `anyhow::Result` in `main.rs`, `candle_core::Result` inside the model module.

## Non-obvious things that bite

- **Never use greedy/argmax sampling with MiniCPM5** — its `<think>` block loops forever on greedy. `main.rs` deliberately uses `Sampling::TopP { p: 0.9, temperature: 0.7 }` with seed `299792458` to break the loop. Decode stops on EOS token ids `[1, 130073]` (from `config.json`) or the max-token limit.
- **`QTensor::quantize_onto` requires a CPU source tensor.** That is why `from_safetensors_dir` mmaps the safetensors on `Device::Cpu` and then quantizes onto Metal. Don't "optimize" this by mmaping directly on Metal.
- The decode loop passes `index_pos = n_prompt + index` and reuses the in-layer KV cache; the mask cache is keyed by `(seq_len, kv_len)` because masks become rectangular once prefix KV entries exist.
- `stream_clean` in `main.rs` strips `<think>`/`</think>` by re-replacing over the accumulated string and tracking a byte offset; the offset only stays valid because clean text grows by complete decoded strings (UTF-8 char boundaries).
- `clear_kv_cache()` exists on `ModelWeights` for reusing a loaded model; the current CLI never calls it (one prompt per process).

## Security considerations

- The binary loads local model/tokenizer files only; it makes no network requests at runtime.
- One `unsafe` block: `VarBuilder::from_mmaped_safetensors` in `from_safetensors_dir` (candle's mmap API). It is sound only as long as the mapped files are not mutated while loaded — do not expose a path that writes to the weights directory while the model is mapped.
- No secrets, credentials, or user data are handled. Prompts come from argv, so they are visible in shell history/process lists — not a concern for local use, but don't add logging of prompts to shared systems.

## Project history (for context)

Recoverable via git stash: `stash@{0}` = original hand-written educational engine (`TinyTensor`, KV cache, ratatui TUI); `stash@{1}` = candle-vllm client (bf16, ~24 tok/s); current = official candle + Q8 + vendored `quantized_minicpm5` (~57 tok/s). Note candle-vllm's fork `--isq` hangs on Metal — this project avoids it by using official candle.

## Related files

- `README.md` — user-facing docs; keep it in sync with any behavior changes.
- `CLAUDE.md` — similar guidance file for Claude Code; when you change commands, architecture, or conventions documented here, update that file too (and vice versa).
