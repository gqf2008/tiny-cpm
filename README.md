# tiny-cpm

A single-binary Rust CLI that runs [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) — a reasoning ("thinking") LLM — **quantized (GGUF, e.g. Q8_0)** on Apple **Metal**, using the official [`candle`](https://github.com/huggingface/candle) crate (0.11, from crates.io) plus a vendored, minimally patched `quantized_minicpm5` model module.

MiniCPM5's `<think>…</think>` reasoning tags are stripped inline during streaming, so the visible output reads as one continuous answer.

Inspired by [tiny-vllm](https://github.com/kuawo/tiny-llm).

## What you get

- ~57–59 tok/s decode on Metal with Q8_0 (vs ~24 tok/s for bf16). Candle's Metal quantized path dequantizes per block then matmuls; for a 1B model the memory-bandwidth win from half-size weights beats the dequant cost.
- Single-shot, synchronous CLI: load → prefill → decode → print → exit. No server, no batching, no streaming protocol — tokens are just printed to stdout as they generate.
- Two load paths from the same binary, auto-detected from `argv[1]`:
  - a pre-converted `.gguf` file → `ModelWeights::from_gguf` (fast load);
  - a bf16 safetensors directory (HF layout: sharded `.safetensors` + `config.json`) → `ModelWeights::from_safetensors_dir`, which mmaps bf16 tensors on CPU and quantizes each to Q8_0 onto Metal at load time (~5 s slower, no llama.cpp needed). Both produce equivalent Q8 weights.

## Prerequisites

macOS needs the Metal Toolchain (one-time):

```bash
xcodebuild -downloadComponent MetalToolchain
```

You also need MiniCPM5-1B weights plus `tokenizer.json`, either as:

- a **GGUF** file (e.g. `MiniCPM5-1B-Q8_0.gguf`, converted with [llama.cpp](https://github.com/ggerganov/llama.cpp)); or
- a **bf16 safetensors directory** in HuggingFace layout (sharded `.safetensors` + `config.json`).

Model weights are not in this repo.

## Build and run

```bash
# A GGUF (pre-converted with llama.cpp) — fast load
cargo run --release -- ./models/MiniCPM5-1B-Q8_0.gguf ./models/tokenizer.json "What is artificial intelligence?" 512

# Or a bf16 safetensors directory — quantizes to Q8_0 in memory at load (~5s slower)
cargo run --release -- ./models/minicpm5-1b/ ./models/tokenizer.json "What is artificial intelligence?" 512
```

The CLI is positional (parsed in `main()` in `src/main.rs`):

| argv  | meaning                            | required               |
|-------|------------------------------------|------------------------|
| `[1]` | `.gguf` file **or** bf16 directory | yes                    |
| `[2]` | `tokenizer.json`                   | yes                    |
| `[3]` | prompt                             | yes                    |
| `[4]` | max tokens to generate             | no (default 512)       |

`argv[1]` is auto-detected: a path ending in `.gguf` loads pre-quantized weights via `ModelWeights::from_gguf`; any other path is treated as a directory and loaded via `ModelWeights::from_safetensors_dir`, which mmaps the bf16 tensors on CPU (`QTensor::quantize_onto` requires a CPU source) and quantizes each onto Metal.

Other commands:

```bash
cargo build --release   # build without running
cargo check             # fast type-check
cargo fmt
cargo clippy
cargo test              # CPU-only mask unit tests in src/quantized_minicpm5.rs
```

## Output and timing

Generated text goes to **stdout**; all diagnostics go to **stderr**. Keep it that way — the stdout stream is the "clean answer" consumers may pipe.

```
quantizing bf16 safetensors in ./models/minicpm5-1b/ to Q8_0 in-memory...   (stderr)
loaded model in 5.12s                                                     (stderr)
prompt tokens: 14                                                         (stderr)
TTFT (prefill 14 tokens + first token): 0.083s                            (stderr)
... streamed answer ...                                                   (stdout)
decode: 480 tokens in 8.14s (59.0 tok/s)                                  (stderr)
```

The `<think>` / `</think>` tags are stripped inline by `stream_clean` in `src/main.rs`: it re-replaces over the accumulated string and tracks a byte offset of what has already been printed (the offset stays valid because clean text grows only by complete decoded strings, at UTF-8 char boundaries).

## Sampling

`main.rs` uses top-p sampling (`p: 0.9`, `temperature: 0.7`, seed `299792458`) via candle's `LogitsProcessor`. Greedy / argmax decoding is deliberately **not** used: MiniCPM5's `<think>` block loops forever on greedy — top-p breaks the loop. Decode stops on either EOS token id (`[1, 130073]`, from `config.json`) or the max-token limit.

## How it works

Three source files, no submodules:

- **`src/main.rs`** — thin driver. Parses CLI args → picks the load path by extension → renders MiniCPM5's ChatML template (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`) → prefill → top-p decode loop → streams tokens with reasoning tags stripped → prints timing. All model math lives in the vendored module, not here. The decode loop passes `index_pos = n_prompt + index` and reuses the in-layer KV cache; `Device::new_metal(0)` is unconditional (macOS / Apple Silicon only).
- **`src/quantized_minicpm5.rs`** — the model. A vendored copy of `candle_transformers::models::quantized_llama` (0.11) with two patches so MiniCPM5's non-standard attention geometry works (see below). Supports 2/3/4/8-bit GGUF (Q8_0, Q4_K_M, …), GQA, MoE (`MlpOrMoe`), an in-layer KV cache, and a mask cache keyed by `(seq_len, kv_len)` (masks become rectangular once prefix KV entries exist). `MAX_SEQ_LEN = 4096`. Also contains the project's only tests: 6 CPU-only unit tests verifying causal-mask shapes, values, and broadcast compatibility.
- **`src/token_output_stream.rs`** — vendored copy of candle's streaming tokenizer wrapper; decodes tokens incrementally so partial words aren't flushed mid-token. Kept identical to upstream.

### The two MiniCPM5 patches (why the model is vendored)

Upstream `quantized_llama` assumes `head_dim == hidden_size / num_heads` and `num_heads * head_dim == hidden_size`. MiniCPM5 has `head_dim = 128`, `hidden_size = 1536`, `num_heads = 16`, so `num_heads * head_dim = 2048 ≠ 1536`. Both assumptions break. The vendored module:

1. Reads `head_dim` from GGUF metadata (`llama.rope.dimension_count`) instead of `embedding_length / head_count` — or from `config.json`'s `head_dim` field on the safetensors path.
2. Reshapes the attention output to `num_heads * head_dim` (2048) before the output projection, which maps 2048 → 1536 (`hidden`).

It also detects the RoPE convention from `general.architecture` on the GGUF path (NEOX for qwen/phi/falcon/…, NORM/interleaved otherwise, matching llama.cpp's `llama_model_rope_type()`); MiniCPM5 uses NORM. The safetensors path hardcodes NORM.

Keeping both vendored files a minimal diff from upstream makes it easy to re-vendor when candle releases updates.

## Tech stack

- Rust, edition 2024 (builds with stable rustc ≥ 1.85-ish — verified with 1.97).
- `candle-core` / `candle-nn` 0.11 with the `metal` feature; `candle-transformers` 0.11.
- `anyhow` (errors in `main.rs`), `serde` + `serde_json` (`config.json` parsing), `tokenizers` 0.22, `tracing`.
- Target platform: macOS / Apple Silicon only.

## License

MIT — see [LICENSE](LICENSE).
