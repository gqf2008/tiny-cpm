# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tiny-cpm` is a single-binary Rust CLI for **on-device inference on Apple Metal** (macOS / Apple Silicon only), built on the **official `candle` 0.11** crate. One binary, six models, six subcommands:

- `chat` — MiniCPM5-1B reasoning LLM, quantized GGUF Q8_0 (~57 tok/s on Metal) via a vendored `quantized_minicpm5` module.
- `asr funasr` / `asr qwen3` — Fun-ASR-Nano-2512 and Qwen3-ASR speech recognition.
- `tts voxcpm` / `tts moss` — VoxCPM2 and MOSS-TTS-Nano (+ MOSS-Audio-Tokenizer-Nano codec) speech synthesis, with `--ref` voice cloning.
- `tts cosyvoice3` — Fun-CosyVoice3-0.5B (Qwen2-0.5B LM + DiT-CFM + HiFT, 24kHz; ported from CrispASR C++/ggml). 8 baked voices (`--voice`), zero-shot cloning (`--ref` + `--ref-text`, required).
- `dialogue` — one-process pipeline: Fun-ASR transcribes input WAV → MiniCPM5 replies → MOSS speaks it; per-stage latency on stderr.
- `vad` — FireRedVAD-Stream-VAD speech-segment detection (safetensors + cmvn.json; `.pth.tar` rejected). stdout: one `start end` (seconds) per segment.
- `live` — realtime voice dialogue: mic (cpal) → FireRedVAD endpointing → Qwen3-ASR → MiniCPM5 (sentence-streamed) → MOSS-TTS per sentence → speaker. `--input/--output` = mic-less simulation mode. No barge-in in v1. Audio IO in `src/utils/live_audio.rs`.

The four speech models are **ported from [`aha`](https://github.com/jhqxxx/aha)** (candle 0.9.2 → 0.11; server/rocket/tokio/modelscope/minijinja coupling stripped). Ported files carry `//! Ported from aha …` headers and keep aha's names/signatures for mechanical re-porting.

**History** — earlier iterations recoverable via git stash: `stash@{0}` hand-written educational engine (`TinyTensor`, ratatui TUI); `stash@{1}` candle-vllm client (bf16, ~24 tok/s); then official candle + Q8 MiniCPM5 chat; the speech models came in on branch `port-asr-tts-models`.

## Commands

```bash
cargo run --release -- chat <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens]
cargo run --release -- asr funasr <model-dir> <audio-file> [max_tokens]
cargo run --release -- asr qwen3  <model-dir> <audio-file> [max_tokens]
cargo run --release -- tts voxcpm <model-dir> "<text>" <out.wav> [--ref ref.wav] [--max-len N]
cargo run --release -- tts moss   <model-dir> "<text>" <out.wav> [--codec dir] [--ref ref.wav] [--max-len N]
cargo run --release -- dialogue <funasr-dir> <minicpm5.gguf|bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> <input.wav> <output.wav> [max_tokens]

cargo build --release
cargo check
cargo fmt
cargo clippy
cargo test    # CPU-only unit tests (masks, feature-length math, config parsing)
```

Output contract: payload (chat/transcript) → **stdout**; diagnostics → **stderr**; TTS writes WAV to the given path.

Model weights are not in the repo — download into `./models/` (gitignored). MOSS models live on ModelScope (`openmoss/…`), the rest on HuggingFace.

## Architecture

- **`src/main.rs`** — subcommand dispatch only.
- **`src/exec/`** — thin per-model CLI drivers (`chat.rs`, `fun_asr_nano.rs`, `qwen3_asr.rs`, `voxcpm.rs`, `moss_tts.rs`, `dialogue.rs`, `live.rs`): parse args → load weights → infer → emit. Model math lives in `src/models/`. Reusable engines also live here: `FunAsrEngine`, `Qwen3AsrEngine` (`transcribe_samples` for in-memory audio), `MossEngine` (`synthesize_pcm` for in-memory playback).
- **`src/quantized_minicpm5.rs`** + **`src/token_output_stream.rs`** — MiniCPM5 chat path, vendored from candle (`quantized_llama` + streaming tokenizer wrapper). Minimal diff from upstream.
- **`src/models/`** — aha ports: `fun_asr_nano/`, `qwen3_asr/`, `voxcpm/`, `moss_tts_nano/`, `moss_audio_tokenizer_nano/`; shared backbones `qwen3/`, `gpt2/`, `feature_extractor/`.
- **`src/common/`** (`modules.rs`, `sample.rs`, `InferenceModel`/`MultiModalData`), **`src/utils/`** (`audio_utils.rs`, `tensor_utils.rs`), **`src/position_embed/`**, **`src/tokenizer/`** — shared infra ported from aha with identical names.

## Non-obvious things that bite

- **`from_gguf` vs `from_safetensors_dir`** (chat): `.gguf` → pre-quantized fast load; a directory → bf16 safetensors quantized to Q8_0 in memory at load. **`quantize_onto` needs a CPU source** — mmap on CPU, then quantize onto Metal.
- **Why vendored `quantized_minicpm5`**: upstream `quantized_llama` hardcodes `head_dim = hidden/heads` (96 for MiniCPM5); MiniCPM5 has `head_dim=128`, `16*128=2048 ≠ 1536`. Patches: read `head_dim` from GGUF metadata / config.json; reshape attention output to `n_head*head_dim` before o_proj.
- **Greedy (`ArgMax`) loops forever** on MiniCPM5 — `chat` uses `TopP { p: 0.9, temperature: 0.7 }`, seed `299792458`, EOS `[1, 130073]`.
- **Weight formats differ per model**: Fun-ASR = `.pt` pickles + bundled `Qwen3-0.6B/` subdir; Qwen3-ASR = mmaped safetensors (bf16); VoxCPM2 = safetensors LM + `.pth` AudioVAE (F32); MOSS = `.bin` pickles (loaded at **F32** — F16 visibly degraded quality vs the official F32 Python reference) + safetensors codec + sentencepiece `tokenizer.model`.
- **MOSS**: `--codec` defaults to `<model-dir>/../MOSS-Audio-Tokenizer-Nano`; `--max-len` default 100 frames (~8 s at 12.5 fps — the codec's true rate); output WAV is stereo.
- **Metal-risky ops**: `conv_transpose1d` (VoxCPM AudioVAE), per-frame `arg_sort`/`to_scalar` syncs (MOSS, perf only). Compile-clean; watch at first real run.
- **ASR prompt templates are rendered manually** (no minijinja): Qwen3-ASR expands `<|audio_pad|>` placeholders; Fun-ASR splices adaptor outputs at `fbank_mask` positions.
- **macOS requires the Metal Toolchain** (one-time): `xcodebuild -downloadComponent MetalToolchain`.

## Code conventions

- Vendored/ported files stay minimal diffs (candle upstream / aha); don't restyle them, keep original comments (including Chinese ones).
- `src/main.rs` and `src/exec/*` stay thin drivers.
- Errors: `anyhow::Result` in drivers; ported aha code keeps `anyhow::Result` (verbatim-port wins); `candle_core::Result` only in the vendored candle files.
- Keep `AGENTS.md` and this file in sync.
