# tiny-cpm

Runs [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) **quantized (GGUF, e.g. Q8_0)** on Apple **Metal** via the official [`candle`](https://github.com/huggingface/candle) Rust crate, with a small vendored `quantized_minicpm5` model.

MiniCPM5 is a reasoning (thinking) model — this prints the `<think>` reasoning and the final answer separately.

Inspired by [tiny-vllm](https://github.com/kuawo/tiny-llm).

## Prerequisites

macOS needs the Metal Toolchain (one-time):

```bash
xcodebuild -downloadComponent MetalToolchain
```

You also need a MiniCPM5-1B GGUF file (e.g. `MiniCPM5-1B-Q8_0.gguf`, converted with [llama.cpp](https://github.com/ggerganov/llama.cpp)) and its `tokenizer.json`.

## Build and run

```bash
# A GGUF (pre-converted with llama.cpp) — fast load
cargo run --release -- ./models/MiniCPM5-1B-Q8_0.gguf ./models/tokenizer.json "What is artificial intelligence?" 512

# Or a bf16 safetensors directory (HF layout + config.json) — no llama.cpp needed,
# quantizes to Q8_0 in memory at load time
cargo run --release -- ./models/minicpm5-1b/ ./models/tokenizer.json "What is artificial intelligence?" 512
```

`argv[1]` is auto-detected: a `.gguf` file loads pre-quantized weights; a directory loads bf16 safetensors and quantizes them to Q8_0 in memory. Both decode at ~57-59 tok/s on Metal.

## How it works

- `src/quantized_minicpm5.rs` — a copy of candle's `quantized_llama` with two patches so MiniCPM5's non-standard `head_dim` (128 ≠ hidden/heads = 96) works.
- `src/main.rs` — loads the GGUF, applies MiniCPM5's ChatML template, decodes with top-p sampling until EOS, and splits `<think>...</think>` from the answer.

## License

MIT — see [LICENSE](LICENSE)
