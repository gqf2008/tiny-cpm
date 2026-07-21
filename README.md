# tiny-cpm

Runs [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) **quantized (GGUF, e.g. Q8_0)** on Apple **Metal** via the official [`candle`](https://github.com/huggingface/candle) Rust crate (0.11, from crates.io), with a small vendored `quantized_minicpm5` model.

MiniCPM5 is a reasoning ("thinking") model. This project streams its output to stdout, stripping the `<think>…</think>` reasoning tags inline so the visible text reads as one continuous answer.

Inspired by [tiny-vllm](https://github.com/kuawo/tiny-llm).

## What you get

- ~57–59 tok/s decode on Metal (Q8_0), versus ~24 tok/s for bf16 — candle's metal quantized path dequantizes per block then matmuls; for a 1B model the memory-bandwidth win from half-size weights beats the dequant cost.
- Single-shot, synchronous CLI: load → prefill → decode → print → exit. No streaming server, no batching.
- Two load paths from the same binary: a pre-converted `.gguf` (fast load) or a bf16 safetensors directory quantized to Q8_0 in memory at load time (no llama.cpp needed).

## Prerequisites

macOS needs the Metal Toolchain (one-time):

```bash
xcodebuild -downloadComponent MetalToolchain
```

You also need MiniCPM5-1B weights plus its `tokenizer.json`. Either:

- a **GGUF** file (e.g. `MiniCPM5-1B-Q8_0.gguf`, converted with [llama.cpp](https://github.com/ggerganov/llama.cpp)); or
- a **bf16 safetensors directory** in HuggingFace layout (sharded `.safetensors` + `config.json`).

## Build and run

```bash
# A GGUF (pre-converted with llama.cpp) — fast load
cargo run --release -- ./models/MiniCPM5-1B-Q8_0.gguf ./models/tokenizer.json "What is artificial intelligence?" 512

# Or a bf16 safetensors directory — quantizes to Q8_0 in memory at load (~5s slower)
cargo run --release -- ./models/minicpm5-1b/ ./models/tokenizer.json "What is artificial intelligence?" 512
```

The CLI is positional:

| argv    | meaning                              | required |
|---------|--------------------------------------|----------|
| `[1]`   | `.gguf` file **or** bf16 directory   | yes      |
| `[2]`   | `tokenizer.json`                    | yes      |
| `[3]`   | prompt                               | yes      |
| `[4]`   | max tokens to generate               | no (default 512) |

`argv[1]` is auto-detected: a path ending in `.gguf` loads pre-quantized weights via `ModelWeights::from_gguf`; any other path is treated as a directory and loaded via `ModelWeights::from_safetensors_dir`, which mmaps the bf16 tensors on CPU and quantizes each onto Metal with `QTensor::quantize_onto`. Both produce equivalent Q8 weights.

Other commands:

```bash
cargo build --release   # build without running
cargo check              # fast type-check
cargo fmt
cargo clippy
cargo test              # small mask-shape tests in quantized_minicpm5.rs
```

## Output and timing

Tokens are streamed to **stdout** as they generate (the `<think>` / `</think>` tags are stripped inline by `stream_clean`, so reasoning and answer come out as one continuous stream rather than two separate blocks). Diagnostics go to **stderr**:

```
quantizing bf16 safetensors in ./models/minicpm5-1b/ to Q8_0 in-memory...
loaded model in 5.12s
prompt tokens: 14
TTFT (prefill 14 tokens + first token): 0.083s
... streamed answer ...
decode: 480 tokens in 8.14s (59.0 tok/s)
```

## Sampling

`main.rs` uses top-p sampling (`p: 0.9`, `temperature: 0.7`, seed `299792458`). Greedy / argmax decoding is **not** used because MiniCPM5's `<think>` block loops on greedy sampling — top-p breaks the loop. Decode stops on either EOS token (`[1, 130073]`, from `config.json`) or the max-token limit.

## How it works

Three source files:

- **`src/main.rs`** — thin driver. Parses CLI args → picks the load path by extension → renders MiniCPM5's ChatML template (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`) → prefill → top-p decode loop → streams tokens with reasoning tags stripped → prints timing.
- **`src/quantized_minicpm5.rs`** — the model. A vendored copy of `candle_transformers::models::quantized_llama` (0.11) with two patches so MiniCPM5's non-standard attention geometry works (see below). Supports 2/3/4/8-bit GGUF (Q8_0, Q4_K_M, …), GQA, an in-layer KV cache, and a mask cache keyed by `(seq_len, kv_len)`. `MAX_SEQ_LEN = 4096`.
- **`src/token_output_stream.rs`** — vendored copy of candle's streaming tokenizer wrapper; decodes tokens incrementally so partial words aren't flushed mid-token.

### The two MiniCPM5 patches

Upstream `quantized_llama` assumes `head_dim == hidden_size / num_heads` and `num_heads * head_dim == hidden_size`. MiniCPM5 has `head_dim = 128` with `hidden_size = 1536`, `num_heads = 16`, so `num_heads * head_dim = 2048 ≠ 1536`. Both assumptions break. The vendored module:

1. Reads `head_dim` from the GGUF metadata (`llama.rope.dimension_count`) instead of `embedding_length / head_count` — or from `config.json`'s `head_dim` field on the safetensors path.
2. Reshapes the attention output to `num_heads * head_dim` (2048) before the output projection, which then maps 2048 → 1536 (`hidden`).

The module also detects the RoPE convention from `general.architecture` for the GGUF path (NEOX for qwen/phi/falcon/…, NORM/interleaved otherwise); MiniCPM5 uses NORM. Keeping this file a minimal diff from upstream makes it easy to re-vendor on candle updates.

## License

MIT — see [LICENSE](LICENSE).
