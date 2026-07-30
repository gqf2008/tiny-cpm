# AGENTS.md

Guidance for AI coding agents working on this repository. Assumes no prior knowledge of the project.

## Project overview

`tiny-cpm` is a single-binary Rust CLI for **on-device inference on Apple Metal** (macOS / Apple Silicon only), built on the official [`candle`](https://github.com/huggingface/candle) crate (0.11, from crates.io). One binary, six models, six subcommands:

- `chat` — [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) reasoning LLM, **quantized (GGUF Q8_0)** via a vendored, minimally patched `quantized_minicpm5` module. ~57–59 tok/s decode on Metal.
- `asr funasr` — Fun-ASR-Nano-2512 (SANM encoder + adaptor + Qwen3-0.6B decoder; `.pt` pickle weights).
- `asr qwen3` — Qwen3-ASR-0.6B/1.7B (Whisper-style audio encoder + Qwen3 decoder; safetensors bf16).
- `tts voxcpm` — VoxCPM2 (MiniCPM4 LM + residual LM + locDiT flow matching + AudioVAE; safetensors + `.pth` VAE). Voice cloning via `--ref`.
- `tts cosyvoice3` — Fun-CosyVoice3-0.5B (Qwen2-0.5B AR speech-token LM + DiT-CFM flow + HiFT vocoder, 24kHz; ported from [CrispASR](https://github.com/CrispStrobe/CrispASR)'s C++/ggml implementation). 8 baked voices via `--voice`, zero-shot cloning via `--ref` + `--ref-text` (required), uses s3tok/CAMPPlus GGUF encoders. Prefers `cosyvoice3-llm-q4_k.gguf` (QMatMul backbone, F32 activations — candle's Metal quantized kernels are F32-only) and `cosyvoice3-flow-q8_0.gguf` (dequantized to F16, attention upcast to F32 to avoid q·k overflow) over `llm.pt`/`flow.pt` when present. `--stream` = chunked streaming (first audio ~1.1 s warm (default first-chunk: hop 12, 3 steps, no CFG; upstream schedule after; env knobs CV3_FIRST_HOP/CV3_FIRST_STEPS/CV3_FIRST_CFG)).
- `tts moss` — MOSS-TTS-Nano (GPT-2-style codec-LM; `.bin` pickles + sentencepiece) + MOSS-Audio-Tokenizer-Nano codec (safetensors). Voice cloning via `--ref`.
- `tts qwen3` — Qwen3-TTS-12Hz-1.7B-Base (ported from [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) `qwen_tts`, **not** aha). Pipeline: Qwen2 BPE → talker (28-layer Qwen3 decoder, dual text+codec embedding) emits codec codebook 0 per 12.5 Hz frame → 5-layer code predictor fills codebooks 1–15 → 16 codes → custom Mimi-family speech-tokenizer decoder → 24 kHz mono. Zero-shot voice cloning via `--ref <ref.wav> --ref-text "<text>"` (speaker encoder ECAPA-TDNN + codec encoder ICL prefix). `--ref-text` must be the *full* ref transcript or the model babbles the ref. Weights: `model.safetensors` BF16 (talker+predictor+speaker-encoder), `speech_tokenizer/model.safetensors` F32 (codec). RTF ~1.4–2.1 on Metal. `--stream` = chunked streaming output (first audio ~1.6–2.5 s): per-frame `generate_stream` callback → each chunk codec-decoded as a sliding window with 25-frame left context trimmed. Fidelity: small chunks approximate the batch PCM near seams (the 25-frame context < codec receptive field; max\|Δ\| 1.2 @ chunk=25 → 0.0 @ chunk=300 bit-identical; guard `tests/stream_decode_equiv.rs`). Tune `--stream-first`/`--stream-chunk`, self-check `QWEN3_TTS_STREAM_CHECK=1`.
- `dialogue` — one-process voice pipeline: Fun-ASR-Nano transcribes an input WAV → MiniCPM5 replies → MOSS-TTS speaks the reply; per-stage latency summary on stderr. Diagnostic env probes: `DIALOGUE_PROBE=1` (MOSS steady-state ms/frame after each stage), `DIALOGUE_PROBE_ONLY=1` + `PROBE_NO_FUNASR`/`PROBE_NO_LLM` (residency isolation).
- `vad` — FireRedVAD-Stream-VAD speech-segment detection (safetensors + `cmvn.json`; torch `.pth.tar` checkpoints rejected). stdout prints one `start end` (seconds) per segment; the streaming `detect_frame*` API is ported and ready for a realtime loop.
- `live` — realtime voice dialogue: mic (cpal/CoreAudio) → FireRedVAD endpointing → Qwen3-ASR → MiniCPM5 (sentence-streamed) → MOSS-TTS per sentence → speaker. `--input <wav>`/`--output <wav>` give a simulation mode for testing without a mic. `--ref <wav>` clones a voice (ref encoded once, reused per sentence). Default half-duplex (ducking). `--barge-in` keeps the mic live during playback and cancels the in-flight reply on speech onset (headphones required — no AEC; env knobs `LIVE_BARGE_RMS`/`LIVE_BARGE_ONSET_FRAMES`).

The four speech models are **ported from [`aha`](https://github.com/jhqxxx/aha)** (candle 0.9.2 → 0.11; rocket server / tokio / modelscope / minijinja coupling stripped; Metal-only). Ported files carry a `//! Ported from aha …` header and keep aha's names/signatures so future aha updates can be re-ported mechanically.

Key characteristics:

- **Single-shot, synchronous CLI**: load → infer → emit → exit. No server, no batching.
- Output contract: payload (chat text / transcript) → **stdout**; diagnostics (load time, TTFT, tok/s) → **stderr**; TTS writes a WAV file. Keep stdout clean — consumers may pipe it.

## Tech stack

- Rust, edition 2024 (stable rustc ≥ 1.85-ish — verified with 1.97). **macOS / Apple Silicon**: `Device::new_metal(0)` is the default everywhere (moss/voxcpm allow a `TINY_CPM_DEVICE=cpu` override); candle built with the `metal` feature.
- Dependencies (`Cargo.toml`): candle-* 0.11, `anyhow`, `serde` + `serde_json` + `serde_yaml` (Fun-ASR `config.yaml`), `tokenizers` 0.22, `sentencepiece` 0.13 (MOSS), `cpal` 0.16 (live mic/speaker), `hound` (WAV), `symphonia` 0.5 (mp3/wav), `realfft`/`rustfft` (STFT/mel), `tracing`, `rand`, `half`, `byteorder`.
- Model weights are **not** in the repo. Download into `./models/` (gitignored). MOSS models are on ModelScope (`openmoss/…`), the rest on HuggingFace.

## Prerequisites

```bash
xcodebuild -downloadComponent MetalToolchain   # one-time
```

## Build, run, and test commands

```bash
cargo run --release -- chat <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens]
cargo run --release -- asr funasr <model-dir> <audio-file> [max_tokens]
cargo run --release -- asr qwen3  <model-dir> <audio-file> [max_tokens]
cargo run --release -- tts voxcpm <model-dir> "<text>" <out.wav> [--ref ref.wav] [--max-len N]
cargo run --release -- tts moss   <model-dir> "<text>" <out.wav> [--codec dir] [--ref ref.wav] [--max-len N]
cargo run --release -- tts cosyvoice3 <model-dir> "<text>" <out.wav> [--voice name] [--ref ref.wav --ref-text "<text>"] [--steps 6] [--stream]
cargo run --release -- tts qwen3 <model-dir> "<text>" <out.wav> [--ref ref.wav --ref-text "<text>"] [--language <lang>] [--max-frames N] [--talker-quant q4_k|q8_0|none] [--stream]
cargo run --release -- dialogue <funasr-dir> <minicpm5.gguf|bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> <input.wav> <output.wav> [max_tokens]
cargo run --release -- live <vad-dir> <qwen3asr-dir> <minicpm5.gguf|bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> [--input <wav>] [--output <wav>] [--max-tokens N] [--ref <wav>]
cargo run --release -- vad <model-dir> <audio-file>

cargo build --release
cargo check
cargo fmt
cargo clippy
cargo test    # CPU-only unit tests
```

## Code organization

- **`src/main.rs`** — subcommand dispatch only (`chat` / `asr` / `tts` / `dialogue` / `vad` / `live`).
- **`src/exec/`** — per-model CLI drivers (`chat.rs`, `fun_asr_nano.rs`, `qwen3_asr.rs`, `voxcpm.rs`, `moss_tts.rs`, `qwen3_tts.rs`, `dialogue.rs`, `live.rs`). Thin: parse args → load weights → run inference → emit. All model math lives in `src/models/`. Reusable engines also live here: `FunAsrEngine`, `Qwen3AsrEngine` (file + in-memory `transcribe_samples`), `MossEngine` (`synthesize` to wav, `synthesize_pcm` to memory), `Qwen3TtsEngine` (`synthesize_pcm`, `encode_ref` for a reusable `RefVoice`).
- **`src/quantized_minicpm5.rs`** + **`src/token_output_stream.rs`** — the MiniCPM5 chat path (vendored from candle, see "two patches" below). Kept a minimal diff from upstream.
- **`src/models/`** — aha ports: `fun_asr_nano/`, `qwen3_asr/`, `voxcpm/`, `moss_tts_nano/`, `moss_audio_tokenizer_nano/`; **`qwen3_tts/`** (ported from QwenLM/Qwen3-TTS, not aha): `config.rs` (serde configs), `talker.rs` (LM + code predictor, reuses `qwen3::Qwen3DecoderLayer`), `quantized_talker.rs` (QMatMul Qwen3 backbone mirror, opt-in), `codec.rs` (Mimi encoder + custom decoder), `speaker_encoder.rs` (ECAPA-TDNN); plus shared backbones `qwen3/`, `gpt2/`, `feature_extractor/` (whisper mel frontend).
- **`src/common/`** — `modules.rs` (attention/MLP/conv builders), `sample.rs` (temp/top-k/top-p/repetition penalty), `InferenceModel` trait + `MultiModalData`.
- **`src/utils/`** — `audio_utils.rs` (load/resample/mel/kaldi-fbank/LFR/STFT/WAV), `live_audio.rs` (cpal mic capture → 16kHz/400-sample frames, speaker playback queue), `tensor_utils.rs` (masks, scatter, linspace).
- **`src/position_embed/`** — RoPE variants, sinusoidal PE. **`src/tokenizer/`** — tokenizers + sentencepiece wrappers.

### The two MiniCPM5 patches (why `quantized_minicpm5` is vendored)

Upstream `quantized_llama` assumes `head_dim == hidden_size / num_heads` and `num_heads * head_dim == hidden_size`. MiniCPM5: `head_dim = 128`, `hidden = 1536`, `heads = 16` → `16*128 = 2048 ≠ 1536`. The vendored module (1) reads `head_dim` from GGUF `llama.rope.dimension_count` / `config.json`, (2) reshapes attention output to `num_heads * head_dim` before the output projection (2048 → 1536). RoPE convention from `general.architecture` (MiniCPM5 = NORM).

## Testing strategy

- `cargo test` runs CPU-only unit tests: causal-mask tests in `quantized_minicpm5.rs`, plus config-parsing / feature-length / splice-shape tests added during the aha ports.
- No CI. Real verification is empirical: run the binary against actual weights in `./models/` and check output quality.

## Code conventions

- **Vendored files stay minimal diffs**: `quantized_minicpm5.rs` / `token_output_stream.rs` vs upstream candle; `src/models/`, `src/common/`, `src/utils/`, `src/position_embed/`, `src/tokenizer/` vs aha. Don't restyle ported code; keep aha's comments (including Chinese ones).
- **`src/main.rs` and `src/exec/*` stay thin drivers**; model math belongs in `src/models/`.
- Comments/docs in English except verbatim ported aha comments. Module-level `//!` docs explain *why*.
- Errors: `anyhow::Result` in drivers (`src/exec/`). Ported aha code (`src/models/`, `src/common/`, `src/utils/`, `src/tokenizer/`) keeps aha's `anyhow::Result` — the verbatim-port convention wins over `candle_core::Result` there; only the vendored candle files (`quantized_minicpm5.rs`) use `candle_core::Result`.
- Device: `Device::new_metal(0)` by default everywhere; the moss/voxcpm TTS drivers honor `TINY_CPM_DEVICE=cpu` for comparison runs (CPU forces F32 weights — candle CPU has no BF16 matmul).

## Non-obvious things that bite

- **Never greedy/argmax with MiniCPM5** — its `<think>` loops forever. `chat` uses `Sampling::TopP { p: 0.9, temperature: 0.7 }`, seed `299792458`; EOS ids `[1, 130073]`.
- **`QTensor::quantize_onto` requires a CPU source tensor** — that's why the bf16→Q8 path mmaps on CPU first.
- **Weight formats differ per model**: Fun-ASR = `.pt` pickles (`pickle::read_all_with_key(.., Some("state_dict"))`) + bundled `Qwen3-0.6B/` subdir; Qwen3-ASR = mmaped safetensors; VoxCPM2 = safetensors LM + `.pth` AudioVAE (kept F32); MOSS = `.bin` pickles (loaded at **F32** — the official Python reference runs F32 on CPU and F16 visibly degraded quality on this ~100M model) + safetensors codec + `tokenizer.model` sentencepiece.
- **MOSS defaults**: `--codec` defaults to `<model-dir>/../MOSS-Audio-Tokenizer-Nano`; `--max-len` default 100 codec frames (~8 s at 12.5 fps — the codec's true frame rate is 12.5 Hz at 48 kHz stereo, NOT 25 fps). Output is stereo.
- **Metal-risky ops** (worked at compile time, watch at runtime): `conv_transpose1d` (VoxCPM AudioVAE), per-frame `arg_sort`/`to_scalar` syncs (MOSS sampling, perf only), bf16 convs.
- ASR prompts are rendered manually (no minijinja): Qwen3-ASR expands `<|audio_pad|>` × `get_feat_extract_output_lengths(mel_frames)`; Fun-ASR uses fake tokens replaced by adaptor outputs at `fbank_mask` positions.
- **Qwen3-TTS weight nesting**: conv weights in this checkpoint sit under `.conv.weight`/`.conv.bias` (so `CausalConv::new` does `vb.pp("conv")`); the RVQ `input_proj`/`output_proj` use a bare `weight`. Check tensor names against `config.json`/safetensors headers, not the upstream Python attribute path.
- **`gen` is a Rust 2024 reserved keyword** — can't be a variable/field name (the talker/generation code uses `gen_cfg`). Applies project-wide.
- **Qwen3-TTS speaker encoder stays F32** even though the talker runs BF16: its STFT/mel front-end is F32, and the ECAPA convs would hit an F32→BF16 dtype mismatch otherwise. Cast its output embedding to `talker_dtype` before the ICL prefix.
- **Qwen3-TTS talker quantization is opt-in and BF16 is the default** — `--talker-quant q4_k|q8_0|none` (env `TINY_CPM_QWEN3_TTS_TALKER`), default `none`. It buys **memory (~3 GB→~1 GB)** and is roughly speed-neutral-to-faster on this 1.7B (measured RTF: BF16 1.40, q4_k 1.28, q8_0 1.47 on the 4-sentence test — the Q4_K bandwidth win narrowly beats its dequant overhead here, unlike the doc's earlier "never faster" claim, which was written while the quantized path was misconfigured). `src/models/qwen3_tts/quantized_talker.rs` mirrors the Qwen3 layer (per-head QK RMSNorm, no bias) with 7 QMatMuls/layer, F32 activations, runtime in-memory `QTensor::quantize_onto` (no GGUF — needs a **CPU**-mmaped VarBuilder source). The talker backbone is a `TalkerBackbone` enum (Full/Quant) extracted from `forward_step`.
- **Quantized backbone must NOT apply the final norm** — the `Full` path (`Vec<Qwen3DecoderLayer>`) returns *raw* layer outputs and `Talker::forward_step` applies `self.norm` (then `codec_head`) once. `QuantizedTalkerBackbone::forward` must do the same: returning a *normed* output makes `forward_step` norm it a **second** time (double RMSNorm re-scales the already-normalized hidden), which is exactly what made every quantized/passthrough run babble from frame 1 while the prompt (frame 0) looked fine. Regression guard: `tests/layer_wiring.rs` (one reference `Qwen3DecoderLayer` vs the F32-passthrough mirror must be **bit-exact** in both prefill and single-token decode).
- **Qwen3-TTS Metal benchmark (M4)**: tiny-cpm is the **fastest of three implementations on Metal** — RTF **1.28** (Q4_K) vs qwen3-tts-rs 4.11 (its opts are CUDA-only) vs audio.cpp 7.93 (bottleneck = ggml codec decoder, 82% of wall; audio.cpp's real track is CUDA at RTF 0.13–0.19). Full three-way tables + repro in **`docs/qwen3_tts_metal_bench.md`**. Gotcha: audio.cpp's Base model **truncates at ~0.4 s without `--reference-text`**, which made early runs look spuriously "fast" — always pass the full ref transcript.

## Security considerations

- The binary loads local model/tokenizer/audio files only; no network requests at runtime.
- `unsafe` blocks: `VarBuilder::from_mmaped_safetensors` (candle mmap API) — sound only while mapped files are not mutated.
- No secrets or credentials handled.

## Project history (for context)

Recoverable via git stash: `stash@{0}` = hand-written educational engine (`TinyTensor`, ratatui TUI); `stash@{1}` = candle-vllm client (bf16, ~24 tok/s); then official candle + Q8 MiniCPM5 chat (~57 tok/s); current adds the four aha speech models (branch `port-asr-tts-models`). candle-vllm's fork `--isq` hangs on Metal — avoided by using official candle.

## Related files

- `README.md` — user-facing docs; keep in sync with behavior changes.
- `CLAUDE.md` — mirrors this file for Claude Code; update both together.
