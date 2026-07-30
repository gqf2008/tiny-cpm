# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tiny-cpm` is a single-binary Rust CLI for **on-device inference on Apple Metal** (macOS / Apple Silicon only), built on the **official `candle` 0.11** crate. One binary, six models, six subcommands:

- `chat` — MiniCPM5-1B reasoning LLM, quantized GGUF Q8_0 (~57 tok/s on Metal) via a vendored `quantized_minicpm5` module.
- `asr funasr` / `asr qwen3` — Fun-ASR-Nano-2512 and Qwen3-ASR speech recognition.
- `tts voxcpm` / `tts moss` — VoxCPM2 and MOSS-TTS-Nano (+ MOSS-Audio-Tokenizer-Nano codec) speech synthesis, with `--ref` voice cloning.
- `tts cosyvoice3` — Fun-CosyVoice3-0.5B (Qwen2-0.5B LM + DiT-CFM + HiFT, 24kHz; ported from CrispASR C++/ggml). 8 baked voices (`--voice`), zero-shot cloning (`--ref` + `--ref-text`, required). Prefers `cosyvoice3-llm-q4_k.gguf` (QMatMul) / `cosyvoice3-flow-q8_0.gguf` (F16) over `llm.pt`/`flow.pt` when present. `--stream` = chunked streaming (first audio ~1.1 s warm (default first-chunk: hop 12, 3 steps, no CFG; upstream schedule after; env knobs CV3_FIRST_HOP/CV3_FIRST_STEPS/CV3_FIRST_CFG)).
- `tts qwen3` — Qwen3-TTS-12Hz-1.7B-Base (ported from QwenLM/Qwen3-TTS `qwen_tts`, **not** aha). Pipeline: Qwen2 BPE → talker (28-layer Qwen3 decoder, dual text+codec embedding) emits codec codebook 0 per 12.5 Hz frame → 5-layer code predictor fills codebooks 1–15 → 16 codes → custom Mimi-family speech-tokenizer decoder → 24 kHz mono. Weights: `model.safetensors` **BF16** (talker+predictor+speaker-encoder) + `speech_tokenizer/model.safetensors` **F32** (codec). Default voice, or ICL zero-shot cloning with `--ref <wav> --ref-text "<transcript>"` (ECAPA-TDNN speaker embedding prefix + codec-encoded ref codes prepended; `--ref-text` must be the *full* ref transcript or the model babbles the ref). `--language <lang>` (default `auto`). RTF ~1.4–1.9 on Metal.
- `dialogue` — one-process pipeline: Fun-ASR transcribes input WAV → MiniCPM5 replies → MOSS speaks it; per-stage latency on stderr.
- `vad` — FireRedVAD-Stream-VAD speech-segment detection (safetensors + cmvn.json; `.pth.tar` rejected). stdout: one `start end` (seconds) per segment.
- `live` — realtime voice dialogue: mic (cpal) → FireRedVAD endpointing → Qwen3-ASR → MiniCPM5 (sentence-streamed) → MOSS-TTS per sentence → speaker. `--input/--output` = mic-less simulation mode. `--ref <wav>` clones a voice (ref encoded once and reused per sentence — without it MOSS runs continuation mode, whose default voice drifts slow/odd at temp 1.7). Default = half-duplex (mic ducked during playback, no echo-loop, use headphones). `--barge-in` = mic stays live during playback; speech onset cancels the in-flight reply (LLM decode loop + TTS stream) and clears the speaker queue — **headphones required** (no AEC; speaker echo false-triggers). Tune onset with `LIVE_BARGE_RMS` (default 0.02) / `LIVE_BARGE_ONSET_FRAMES` (default 3). Audio IO in `src/utils/live_audio.rs`.

The four speech models are **ported from [`aha`](https://github.com/jhqxxx/aha)** (candle 0.9.2 → 0.11; server/rocket/tokio/modelscope/minijinja coupling stripped). Ported files carry `//! Ported from aha …` headers and keep aha's names/signatures for mechanical re-porting.

**History** — earlier iterations recoverable via git stash: `stash@{0}` hand-written educational engine (`TinyTensor`, ratatui TUI); `stash@{1}` candle-vllm client (bf16, ~24 tok/s); then official candle + Q8 MiniCPM5 chat; the speech models came in on branch `port-asr-tts-models`.

## Commands

```bash
cargo run --release -- chat <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens]
cargo run --release -- asr funasr <model-dir> <audio-file> [max_tokens]
cargo run --release -- asr qwen3  <model-dir> <audio-file> [max_tokens]
cargo run --release -- tts voxcpm <model-dir> "<text>" <out.wav> [--ref ref.wav] [--max-len N]
cargo run --release -- tts moss   <model-dir> "<text>" <out.wav> [--codec dir] [--ref ref.wav] [--max-len N]
cargo run --release -- tts qwen3  <model-dir> "<text>" <out.wav> [--ref ref.wav --ref-text "<text>"] [--language <lang>] [--max-frames N] [--talker-quant q4_k|q8_0|none]
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
- **`src/exec/`** — thin per-model CLI drivers (`chat.rs`, `fun_asr_nano.rs`, `qwen3_asr.rs`, `voxcpm.rs`, `moss_tts.rs`, `qwen3_tts.rs`, `dialogue.rs`, `live.rs`): parse args → load weights → infer → emit. Model math lives in `src/models/`. Reusable engines also live here: `FunAsrEngine`, `Qwen3AsrEngine` (`transcribe_samples` for in-memory audio), `MossEngine` (`synthesize_pcm` for in-memory playback), `Qwen3TtsEngine` (`synthesize_pcm`, `encode_ref` for a reusable `RefVoice`).
- **`src/quantized_minicpm5.rs`** + **`src/token_output_stream.rs`** — MiniCPM5 chat path, vendored from candle (`quantized_llama` + streaming tokenizer wrapper). Minimal diff from upstream.
- **`src/models/`** — aha ports: `fun_asr_nano/`, `qwen3_asr/`, `voxcpm/`, `moss_tts_nano/`, `moss_audio_tokenizer_nano/`; **`qwen3_tts/`** (ported from QwenLM/Qwen3-TTS, not aha): `config.rs` (serde configs), `talker.rs` (LM + code predictor, reuses `qwen3::Qwen3DecoderLayer`), `codec.rs` (Mimi encoder + custom decoder), `speaker_encoder.rs` (ECAPA-TDNN); shared backbones `qwen3/`, `gpt2/`, `feature_extractor/`.
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
- **Qwen3-TTS weight nesting**: conv weights in this checkpoint sit under `.conv.weight`/`.conv.bias` (so `CausalConv::new` does `vb.pp("conv")`); the RVQ `input_proj`/`output_proj` use a bare `weight`. Check tensor names against `config.json`/safetensors headers, not the upstream Python attribute path.
- **`gen` is a Rust 2024 reserved keyword** — can't be a variable/field name (the talker/generation code uses `gen_cfg`). Applies project-wide.
- **Qwen3-TTS speaker encoder stays F32** even though the talker runs BF16: its STFT/mel front-end is F32, and the ECAPA convs would hit an F32→BF16 dtype mismatch otherwise. Cast its output embedding to `talker_dtype` before the ICL prefix.
- **Qwen3-TTS talker quantization is opt-in and BF16 is the default** — `--talker-quant q4_k|q8_0|none` (env `TINY_CPM_QWEN3_TTS_TALKER`), default `none`. It buys **memory (~3 GB→~1 GB)** and is roughly speed-neutral-to-faster on this 1.7B (measured RTF: BF16 1.40, q4_k 1.28, q8_0 1.47 on the 4-sentence test — the Q4_K bandwidth win narrowly beats its dequant overhead here, unlike the doc's earlier "never faster" claim, which was written while the quantized path was misconfigured). `src/models/qwen3_tts/quantized_talker.rs` mirrors the Qwen3 layer (per-head QK RMSNorm, no bias) with 7 QMatMuls/layer, F32 activations, runtime in-memory `QTensor::quantize_onto` (no GGUF — needs a **CPU**-mmaped VarBuilder source). The talker backbone is a `TalkerBackbone` enum (Full/Quant) extracted from `forward_step`.
- **Quantized backbone must NOT apply the final norm** — the `Full` path (`Vec<Qwen3DecoderLayer>`) returns *raw* layer outputs and `Talker::forward_step` applies `self.norm` (then `codec_head`) once. `QuantizedTalkerBackbone::forward` must do the same: returning a *normed* output makes `forward_step` norm it a **second** time (double RMSNorm re-scales the already-normalized hidden), which is exactly what made every quantized/passthrough run babble from frame 1 while the prompt (frame 0) looked fine. Regression guard: `tests/layer_wiring.rs` (one reference `Qwen3DecoderLayer` vs the F32-passthrough mirror must be **bit-exact** in both prefill and single-token decode).
- **Qwen3-TTS Metal benchmark (M4)**: tiny-cpm is the **fastest of three implementations on Metal** — RTF **1.28** (Q4_K) vs qwen3-tts-rs 4.11 (its opts are CUDA-only) vs audio.cpp 7.93 (bottleneck = ggml codec decoder, 82% of wall; audio.cpp's real track is CUDA at RTF 0.13–0.19). Full three-way tables + repro in **`docs/qwen3_tts_metal_bench.md`**. Gotcha: audio.cpp's Base model **truncates at ~0.4 s without `--reference-text`**, which made early runs look spuriously "fast" — always pass the full ref transcript.

## Code conventions

- Vendored/ported files stay minimal diffs (candle upstream / aha); don't restyle them, keep original comments (including Chinese ones).
- `src/main.rs` and `src/exec/*` stay thin drivers.
- Errors: `anyhow::Result` in drivers; ported aha code keeps `anyhow::Result` (verbatim-port wins); `candle_core::Result` only in the vendored candle files.
- Keep `AGENTS.md` and this file in sync.
