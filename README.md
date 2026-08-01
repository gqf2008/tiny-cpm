# tiny-cpm

**On-device (edge) inference on Apple Metal: a single tiny Rust binary for LLM chat, ASR, and TTS — no server, no cloud, no Python.**

A single-binary Rust CLI built on the official [`candle`](https://github.com/huggingface/candle) crate (0.11, from crates.io), macOS / Apple Silicon only (`Device::new_metal(0)`). Models:

| subcommand | model | task | weights |
|------------|-------|------|---------|
| `chat` | [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | reasoning LLM | GGUF (Q8_0 quantized) or bf16 safetensors dir |
| `asr funasr` | [Fun-ASR-Nano-2512](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) | speech recognition | `.pt` pickles + bundled Qwen3-0.6B |
| `asr qwen3` | [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) / 1.7B | speech recognition | safetensors (bf16) |
| `tts voxcpm` | [VoxCPM2](https://huggingface.co/OpenBMB/VoxCPM2) | TTS + voice cloning | safetensors (bf16) + `.pth` AudioVAE |
| `tts moss` | [MOSS-TTS-Nano](https://modelscope.cn/models/openmoss/MOSS-TTS-Nano) + [MOSS-Audio-Tokenizer-Nano](https://modelscope.cn/models/openmoss/MOSS-Audio-Tokenizer-Nano) | TTS + voice cloning | `.bin` pickles + sentencepiece; codec: safetensors |
| `tts cosyvoice3` | [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | TTS (9 languages + 18 zh dialects) + zero-shot cloning | `.pt` pickles (llm/flow/hift) + GGUF (voices/s3tok/campplus) |
| `vad` | [FireRedVAD-Stream-VAD](https://modelscope.cn/models/jiangjiangaha/FireRedVAD-Stream-VAD) | voice activity detection | safetensors + `cmvn.json` (torch `.pth.tar` checkpoints are rejected) |

The four speech models are ported from [`aha`](https://github.com/jhqxxx/aha) (candle 0.9.2 → 0.11, server framework stripped, Metal-only). CosyVoice3 is ported from [CrispASR](https://github.com/CrispStrobe/CrispASR)'s C++/ggml implementation. MiniCPM5 chat uses a vendored, minimally patched `quantized_minicpm5` module (see below).

A fifth subcommand, `dialogue`, chains three models in one process (all loaded once, kept resident): Fun-ASR-Nano transcribes an input WAV → MiniCPM5 replies → MOSS-TTS speaks the reply, with a per-stage latency summary on stderr.

A sixth subcommand, `live`, is the realtime version: microphone → FireRedVAD endpointing → Qwen3-ASR → MiniCPM5 (reply streams sentence-by-sentence) → MOSS-TTS per sentence → speaker playback (cpal/CoreAudio). `--input <wav>` runs a simulation mode (no mic needed) that feeds a WAV through the same loop and writes the reply audio to `--output <wav>`. v1 has no barge-in.

## Prerequisites

macOS needs the Metal Toolchain (one-time):

```bash
xcodebuild -downloadComponent MetalToolchain
```

Model weights are not in this repo. Download them into `./models/` (gitignored), e.g.:

```bash
hf download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
hf download OpenBMB/VoxCPM2 --local-dir models/VoxCPM2
# MOSS models are on ModelScope:
#   https://modelscope.cn/models/openmoss/MOSS-TTS-Nano  (config.json, pytorch_model.bin, tokenizer.model, ...)
#   https://modelscope.cn/models/openmoss/MOSS-Audio-Tokenizer-Nano  (config.json, *.safetensors)
# Fun-ASR-Nano-2512: HF or ModelScope, FunAudioLLM/Fun-ASR-Nano-2512
```

## Build and run

```bash
# MiniCPM5-1B chat — GGUF (fast load) or bf16 safetensors dir (quantized in memory: default Q8_0; override with `TINY_CPM_QUANT` or `--quant <name>` — q8_0, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k, q3_k, q2_k, f16, f32)
cargo run --release -- chat ./models/MiniCPM5-1B-Q8_0.gguf ./models/tokenizer.json "What is AI?" 512
# bf16 dir, custom in-memory quant level (default Q8_0):
cargo run --release -- chat ./models/MiniCPM5-1B ./models/tokenizer.json "What is AI?" 512 --quant q4_k

# ASR — transcript goes to stdout
cargo run --release -- asr funasr ./models/Fun-ASR-Nano-2512 ./audio.wav
cargo run --release -- asr qwen3 ./models/Qwen3-ASR-0.6B ./audio.wav

# TTS — writes a WAV file; --ref clones the voice from a reference clip
cargo run --release -- tts voxcpm ./models/VoxCPM2 "你好，世界。" out.wav [--ref ref.wav] [--max-len N]
cargo run --release -- tts moss ./models/MOSS-TTS-Nano "你好，世界。" out.wav [--codec ./models/MOSS-Audio-Tokenizer-Nano] [--ref ref.wav] [--max-len N]
cargo run --release -- tts cosyvoice3 ./models/Fun-CosyVoice3-0.5B-2512 "你好，世界。" out.wav [--voice zero_shot] [--ref ref.wav --ref-text "参考音频文本"] [--steps 6] [--stream]
```

Full CLI contract (parsed in `src/main.rs`):

```
tiny-cpm chat <model.gguf | bf16-dir> <tokenizer.json> "<prompt>" [max_tokens] [--quant <name>]
tiny-cpm asr <funasr|qwen3> <model-dir> <audio-file> [max_tokens]
tiny-cpm tts <voxcpm|moss> <model-dir> "<text>" <out.wav> [--codec <codec-dir>] [--ref <ref.wav>] [--max-len N]
tiny-cpm tts cosyvoice3 <model-dir> "<text>" <out.wav> [--voice <name>] [--ref <ref.wav> --ref-text "<text>"] [--steps N] [--max-tokens N] [--stream]
tiny-cpm dialogue <funasr-dir> <minicpm5.gguf | bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> <input.wav> <output.wav> [max_tokens]
tiny-cpm live <vad-dir> <qwen3asr-dir> <minicpm5.gguf | bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> [--input <wav>] [--output <wav>] [--max-tokens N] [--barge-in] [--ref <wav>] [--vad-speech-threshold <f>] [--vad-min-speech-frame <n>] [--vad-min-silence <n>] [--vad-min-speach-ratio <f>] [--vad-min-speach-frames <n>] [--vad-end-silence-ratio <f>] [--vad-look-back-frames <n>] [--min-segment-samples <n>] [--min-seg-peak <f>] [--barge-onset-frames <n>] [--min-barge-rms <f>] [--mic-gain <f>]
tiny-cpm vad <model-dir> <audio-file>
tiny-cpm codec-rt <model-dir> <in.wav> <out.wav> [--codec <codec-dir>]   # MOSS codec encode→decode round-trip (diagnostic)
```

cosyvoice3 `--stream`: chunked streaming synthesis — first audio ~1.1 s warm (first chunk: hop 12, 3 steps, no CFG by default, upstream hop schedule afterwards; env knobs CV3_FIRST_HOP/CV3_FIRST_STEPS/CV3_FIRST_CFG; chunks are buffered and one WAV is written for now).

`live --barge-in`: keeps the mic live during TTS playback and cancels the in-flight reply (LLM decode loop + TTS stream) on speech onset, then processes the interrupting utterance. **Headphones required** — there's no AEC, so speaker echo would false-trigger barge-in. Default (no flag) is half-duplex (mic ducked during playback). Tune onset with `--barge-onset-frames` (default 3).

VAD/tuning flags (priority: flag > env var > default). VAD: `--vad-speech-threshold` (0.5, neural speech-prob threshold), `--vad-min-speech-frame` (8, shortest speech segment), `--vad-min-silence` (45 postprocessor / 32 streaming-endpoint, min trailing silence before endpoint), `--vad-min-speach-ratio` (0.1, per-frame speech fraction), `--vad-min-speach-frames` (30, min speech before endpointing), `--vad-end-silence-ratio` (0.9, silence fraction in look-back to endpoint), `--vad-look-back-frames` (50, endpoint silence window). Live: `--min-segment-samples` (8000 = 0.5s, drop shorter segments), `--min-seg-peak` (0.0 = off, drop distant/bystander speech below this peak), `--barge-onset-frames` (3, consecutive neural-speech frames to barge), `--min-barge-rms` (0.015, mic RMS floor for barge), `--mic-gain` (1.0, linear gain for low-gain mics). All have matching `VAD_*` / `LIVE_*` env vars as fallback.

MOSS `--ref` voice cloning: the codec encode (encoder + RVQ) runs on **CPU + f32**, not Metal — Metal f32 drifted on reference audio longer than ~16 s (async-kernel races in the projected-transformer attention + RVQ residual accumulation), producing garbage conditioning codes; a 22 s ref round-trips to noise on Metal but clean on CPU. The decoder stays on Metal (it only decodes short generated sequences). `codec-rt` round-trips a WAV through encode→decode to verify the codec reproduces the input.

A seventh entry point, the `web` crate, is a WebSocket voice-dialogue server in `web/` (a workspace member). The browser does mic capture via `getUserMedia` (echoCancellation/noiseSuppression/autoGainControl — so **speakers work without headphones**, no AEC needed in our code), TTS playback, **live VAD-param sliders**, and a **persona field** (LLM system prompt). It reuses the same engines; barge-in works with speakers because the browser AEC removes the echo.

```bash
cargo run --release --manifest-path web/Cargo.toml -- \
  --vad-dir models/FireRedVAD-Stream-VAD --asr-dir models/Qwen3-ASR-0.6B \
  --minicpm-dir models/MiniCPM5-1B --tokenizer models/MiniCPM5-1B/tokenizer.json \
  --moss-dir models/MOSS-TTS-Nano --codec-dir models/MOSS-Audio-Tokenizer-Nano \
  --ref bailian-console-aliyun-com.wav [--addr 127.0.0.1:8080]
# open the printed URL in Chrome, grant mic permission
```

tiny-cpm is lib+bin (`src/lib.rs` exposes the modules) so the `web` crate depends on it via path.

Output contract: the payload (chat text / transcript) goes to **stdout**; all diagnostics (load time, TTFT, tok/s) go to **stderr**. TTS writes its WAV to the given path. Keep it that way — stdout is what consumers may pipe.

Other commands:

```bash
cargo build --release
cargo check
cargo fmt
cargo clippy
cargo test    # CPU-only unit tests (masks, feature-length math, config parsing, chat repeat-guard)
```

## How it works

- **`src/main.rs`** — subcommand dispatch only. Drivers live in **`src/exec/`** (`chat.rs`, `fun_asr_nano.rs`, `qwen3_asr.rs`, `voxcpm.rs`, `moss_tts.rs`, `dialogue.rs`, `vad.rs`, `live.rs`): parse args → load weights → run inference → emit payload/diagnostics. Live audio IO (mic/speaker via cpal) is in **`src/utils/live_audio.rs`**.
- **`src/quantized_minicpm5.rs`** + **`src/token_output_stream.rs`** — the original MiniCPM5 path: a vendored copy of `candle_transformers::models::quantized_llama` (0.11) with patches for MiniCPM5's non-standard attention geometry and RoPE convention (see below). `MAX_SEQ_LEN = 4096`.
- **`src/models/`** — the aha ports: `fun_asr_nano/` (SANM encoder + adaptor + Qwen3-0.6B decoder), `qwen3_asr/` (Whisper-style audio encoder + Qwen3 decoder), `voxcpm/` (MiniCPM4 LM + residual LM + locDiT flow matching + AudioVAE), `moss_tts_nano/` (GPT-2-style codec-LM) + `moss_audio_tokenizer_nano/` (LFQ codec), plus shared backbones `qwen3/`, `gpt2/`, `feature_extractor/` (whisper mel frontend).
- **`src/common/`** (`modules.rs`, `sample.rs`, `InferenceModel`/`MultiModalData`), **`src/utils/`** (`audio_utils.rs` — load/resample/mel/fbank/STFT/WAV; `tensor_utils.rs`), **`src/position_embed/`** (RoPE, sinusoidal), **`src/tokenizer/`** — shared infrastructure ported from aha with names/signatures kept identical, so re-porting future aha updates stays mechanical.

### The three MiniCPM5 patches (why `quantized_minicpm5` is vendored)

Upstream `quantized_llama` assumes `head_dim == hidden_size / num_heads` and `num_heads * head_dim == hidden_size`. MiniCPM5 has `head_dim = 128`, `hidden_size = 1536`, `num_heads = 16`, so `num_heads * head_dim = 2048 ≠ 1536`. The vendored module reads `head_dim` from GGUF metadata (`llama.rope.dimension_count`) / `config.json`, and reshapes the attention output to `num_heads * head_dim` (2048) before the output projection maps 2048 → 1536. RoPE is always NEOX (non-interleaved / half-split): real MiniCPM5-1B GGUFs declare `general.architecture = "llama"`, but the HF `LlamaForCausalLM` reference uses `rotate_half` (NEOX), so the arch string would select the wrong convention — the loader hardcodes NEOX and prints the arch at load for diagnostics. The bf16 safetensors loader (`from_safetensors_dir`) quantizes HF-layout weights in memory (default Q8_0; `--quant <name>` or `TINY_CPM_QUANT` to override).

### Chat sampling note

`exec/chat.rs` uses top-p sampling (`p: 0.9`, `temperature: 0.7`, seed `299792458`). Greedy decoding is deliberately avoided: MiniCPM5's `<think>` block loops forever on greedy. `<think>`/`</think>` tags are stripped inline while streaming. Decode stops on EOS ids `[1, 130073]` or the max-token limit.

## Tech stack

- Rust, edition 2024 (stable rustc ≥ 1.85-ish — verified with 1.97). Target: macOS / Apple Silicon only.
- `candle-core` / `candle-nn` / `candle-transformers` 0.11 (`metal` feature).
- Audio: `hound` (WAV), `symphonia` (mp3/wav decode), `realfft`/`rustfft` (STFT/mel).
- Tokenization: `tokenizers` 0.22, `sentencepiece` 0.13 (MOSS).
- `anyhow`, `serde`/`serde_json`/`serde_yaml`, `tracing`, `rand`, `half`, `byteorder`.

## License

MIT — see [LICENSE](LICENSE). Model ports in `src/models/`, `src/common/`, `src/utils/`, `src/position_embed/`, `src/tokenizer/` are derived from [`aha`](https://github.com/jhqxxx/aha) (Apache-2.0).
