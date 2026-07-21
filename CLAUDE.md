# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tiny-cpm` runs [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) **quantized (GGUF, e.g. Q8_0)** on Apple **Metal**, using the **official `candle` 0.11** crate (crates.io, not a fork) plus a vendored `quantized_minicpm5` module. Two source files: `src/main.rs` (CLI + ChatML + decode loop + reasoning split) and `src/quantized_minicpm5.rs` (the model).

**History** — this project went through three iterations, all recoverable via git stash:
- `git stash@{0}` — the original hand-written educational engine (`TinyTensor`, KV cache, ratatui TUI).
- `git stash@{1}` — a candle-vllm client (bf16, ~24 tok/s).
- **current** — official candle + Q8 + vendored `quantized_minicpm5` (~57 tok/s, fastest).

## Commands

```bash
# Build & run. argv[1] is either a GGUF file (pre-converted Q8) or a bf16
# safetensors directory (auto-quantized to Q8_0 in-memory at load).
cargo run --release -- "<model.gguf | dir>" "<tokenizer.json>" "<prompt>" [max_tokens]

# GGUF (pre-converted with llama.cpp) — fast load
cargo run --release -- ./models/MiniCPM5-1B-Q8_0.gguf ./models/tokenizer.json "What is AI?" 512

# Or a bf16 safetensors directory (HF layout + config.json) — no llama.cpp needed,
# quantizes to Q8_0 in memory at load (~5s slower load)
cargo run --release -- ./models/minicpm5-1b/ ./models/tokenizer.json "What is AI?" 512

cargo build --release     # build without running
cargo check               # fast type-check
cargo fmt
cargo clippy
```

CLI is positional, parsed in `main()`:
- `argv[1]` = GGUF model file (required)
- `argv[2]` = tokenizer.json (required)
- `argv[3]` = prompt (required)
- `argv[4]` = max_tokens (optional, default 512)

There is **no test suite**.

## Architecture

- **`src/main.rs`** — CLI parse → load GGUF via `candle_core::quantized::gguf_file` → `ModelWeights::from_gguf` → apply MiniCPM5's ChatML template → prefill → top-p decode loop (stops on eos `[1, 130073]`) → split `<think>...</think>` from the answer → print, with load/TTFT/decode timing on stderr.
- **`src/quantized_minicpm5.rs`** — vendored from `candle_transformers::models::quantized_llama` (0.11) with two patches so MiniCPM5 works (see below).

## Non-obvious things that bite

- **`from_gguf` vs `from_safetensors_dir`**: `main.rs` picks by `argv[1]` — `.gguf` file → `from_gguf` (read pre-quantized weights, fast load); a directory → `from_safetensors_dir` (load bf16 safetensors on CPU, `quantize_onto` each weight to Q8_0 onto Metal, slower load ~5s but needs no llama.cpp). Both produce equivalent Q8 weights (~59 tok/s).
- **`quantize_onto` needs a CPU source**: candle's `QTensor::quantize_onto(src, dtype, dev)` requires `src` on CPU (the quantized storage lands on `dev`). So `from_safetensors_dir` mmaps on `Device::Cpu`, then quantizes onto Metal.
- **Why a vendored `quantized_minicpm5`**: upstream `quantized_llama` hardcodes `head_dim = hidden/heads` (= 96 for MiniCPM5) and reshapes attention output to `hidden` (1536). MiniCPM5 has `head_dim=128`, so `num_heads*head_dim = 2048 ≠ hidden`. Two patches: (1) `head_dim` read from GGUF `llama.rope.dimension_count`; (2) attention output reshaped to `n_head*head_dim` (the o_proj then maps 2048→1536). Documented at the top of the file.
- **Greedy (`ArgMax`) loops forever** on MiniCPM5 — its `<think>` repeats. `main.rs` defaults to `Sampling::TopP { p: 0.9, temperature: 0.7 }` to break the loop.
- **macOS requires the Metal Toolchain** (one-time): `xcodebuild -downloadComponent MetalToolchain`.
- **Q8 on Metal via candle works and is fast** (~57 tok/s, beats bf16's 24). candle's metal quantized path dequantizes per block then matmuls; for a 1B model the memory-bandwidth win from half-size weights exceeds the dequant cost. (candle-vllm's fork `--isq` hangs on Metal — different kernel; this project avoids it by using official candle.)
- **Single-shot, synchronous**: no streaming, no server. Load → prefill → decode → print → exit.

## Code conventions

- Keep `quantized_minicpm5.rs` a **minimal diff** from upstream `quantized_llama` (only the two patches), so it's easy to re-vendor on candle updates.
- `main.rs` is a thin driver; all model math lives in the vendored module.
