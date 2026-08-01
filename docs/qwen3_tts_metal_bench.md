# Qwen3-TTS on Apple M4 / Metal — three-way benchmark

Date: 2026-07-30 (original measurements). Updated: 2026-08-01 — the tiny-cpm
headline numbers now reflect the **post-GPU-sampling** configuration (see
"GPU sampling" below); the pre-GPU-sampling measurements are kept in the
History section. Hardware: Apple M4 (arm64), macOS, Metal backend for all three.

Test sentence (same for all three):

> 你好，这是一段测试。很高兴见到你，今天天气真不错。

Reference voice for the clone runs: `../audio.cpp/assets/resources/b.wav` (14 s, English)
with its transcript as `--reference-text` / `--ref-text`:

> Some call me nature, others call me Mother Nature. I've been here for over four point
> five billion years, twenty-two thousand five hundred times longer than you.

All numbers below are from **clean, serial** runs (nothing else contending for the GPU).
Earlier concurrent runs (two engines at once) inflated every number and were discarded.

## Headline (post-GPU-sampling tiny-cpm, per AGENTS.md)

| Engine | Stack | Output | Wall | **RTF** | Note |
|--------|-------|--------|------|---------|------|
| **tiny-cpm** | candle 0.11, Q4_K talker, **GPU sampling** | — | — | **~0.94** | **sub-realtime** on an idle machine |
| tiny-cpm | candle 0.11, BF16 talker, GPU sampling | — | — | ~1.9 | |
| **audio.cpp** | ggml, optimal Metal build | 6.08 s | 48.20 s | **7.93** | bottleneck = codec decoder |
| qwen3-tts-rs | candle 0.9 | 5.04 s | 20.73 s | 4.11 | opts are CUDA-only |
| qwen3-tts-rs (long) | candle 0.9 | 17.04 s | 80.75 s | 4.74 | |

RTF = wall-clock / audio duration. < 1.0 = faster than realtime.

The post-GPU-sampling tiny-cpm numbers (Q4_K **~0.94**, BF16 **~1.9**) are taken
from AGENTS.md (section "Qwen3-TTS Metal benchmark (M4)"), measured on an idle
machine after the sampling path moved fully on-device (see below). They are not
re-measured here; wall-clock columns are left blank rather than fabricate values
for a different configuration than the original run. Benchmark caveat from
AGENTS.md: RTF is only meaningful on an idle machine — one competing GPU/CPU
process swings identical runs 2–6×; check `sysctl -n vm.loadavg` (~<3) first.

**On Metal, tiny-cpm is the fastest of the three** — sub-realtime Q4_K vs ~8×
slower audio.cpp (7.93/0.94) and ~4× slower qwen3-tts-rs (4.11/0.94) on the
same sentence, same hardware.

## GPU sampling (why the tiny-cpm numbers changed)

The original 2026-07-30 measurements (History below) predate the GPU-sampling
change: codebook 0 and the predictor's codebooks 1–15 were sampled with blocking
GPU→CPU readbacks per step, which drained the command buffer and serialized the
pipeline. Moving sampling fully on-device (Gumbel-max as tensor ops, a single
end-of-frame scalar readback) took Q4_K from RTF 1.28 to **~0.94 (sub-realtime)**
on an idle M4 (BF16 ~1.9) — see AGENTS.md for the full breakdown and the
`gpu_sample_token` docs in `src/models/qwen3_tts/talker.rs`.

### Env knobs

| Knob | Default | Effect |
|------|---------|--------|
| `QWEN3_TTS_CPU_SAMPLE` | off (GPU sampling) | `1`/`true` → per-step CPU sampling reference path (exact top-k/top-p); `0`/`off`/`false` disables |
| `QWEN3_TTS_PREDICTOR_QUANT` | on when the talker is quantized | `0`/`off`/`false`/`none` → keep the code predictor BF16; any other value → quantize it (same quant as the talker) |
| `QWEN3_TTS_GREEDY` | off | `1` → force argmax (`do_sample=false`) to test whether babbling is sampling-driven |
| `QWEN3_TTS_PROF` / `QWEN3_TTS_PROF_SYNC` | off | per-stage wall-clock profiling; `_SYNC` drains the GPU between stages (diagnostic, pessimistic totals) |
| `QWEN3_TTS_DEBUG` / `QWEN3_TTS_TRACE` / `QWEN3_TTS_SPEC_PROBE` | off | per-frame code0 logging / trace / predictor speculatability probe |

All boolean knobs parse values: `=0`/`=off`/`=false`/`=no` disable, `=1`/`=true`
enable, unset falls back to the default above (presence-only checks were removed
so `QWEN3_TTS_CPU_SAMPLE=0` really disables the knob).

## audio.cpp per-stage (clean serial, optimal Metal build)

Build: `-DCMAKE_BUILD_TYPE=Release -DGGML_METAL_NDEBUG=ON -DGGML_METAL_EMBED_LIBRARY=ON -DENGINE_ENABLE_OPENMP=OFF`.

| Stage | Time | % of wall |
|-------|------|-----------|
| talker prefill | 0.47 s | |
| talker code_predictor | 3.21 s | |
| talker cached_step | 3.20 s | |
| **talker total** | **7.20 s** | 15 % |
| **speech_decoder** | **39.44 s** | **82 %** |
| voice_prompt | 1.55 s | 3 % |
| **WALL** | **48.20 s** | 100 % |

audio.cpp's **talker static graph is genuinely fast** (7.2 s — the one stage that beats a
naive per-op dispatch). Its **codec/speech decoder is the bottleneck**: ggml rebuilds a
fresh static graph for the conv-heavy Mimi decoder and runs it as one giant `compute`
(39 s for 6 s of audio). This is inherent to ggml-on-Metal for this op mix, not a build
misconfiguration — Release + `GGML_METAL_NDEBUG=ON` only shaved ~11 % off the
RelWithDebInfo build.

Important context from audio.cpp's own README: **CUDA is the optimized path**; Metal,
Vulkan, and CPU are "intended for portability and testing... performance and model
coverage may be lower." Its real reference numbers (CUDA) are **RTF 0.13–0.19** — a
different track we are not on. On Metal it is not the optimized target.

## History: pre-GPU-sampling measurements (2026-07-30)

These are the original clean, serial runs that produced the old headline (Q4_K
1.28). They predate GPU sampling; keep them for the record and for before/after
comparison. (Post-GPU-sampling: Q4_K ~0.94, BF16 ~1.9 — see Headline.)

| Engine | Stack | Output | Wall | **RTF** |
|--------|-------|--------|------|---------|
| tiny-cpm | candle 0.11, Q4_K talker | 5.36 s | 6.86 s | 1.28 |
| tiny-cpm | candle 0.11, BF16 talker | 4.48 s | 6.26 s | 1.40 |
| tiny-cpm | candle 0.11, Q8_0 talker | 6.64 s | 9.75 s | 1.47 |

### tiny-cpm quantization (pre-GPU-sampling, clean, default voice, same sentence)

| Talker | Audio | Wall | RTF |
|--------|-------|------|-----|
| BF16 (no quant) | 4.48 s | 6.26 s | 1.40 |
| **Q4_K** | 5.36 s | 6.86 s | **1.28** |
| Q8_0 | 6.64 s | 9.75 s | 1.47 |

Q4_K wins on this 1.7 B talker: the memory-bandwidth saving narrowly beats the dequant
overhead (AGENTS.md confirms this still holds post-GPU-sampling — Q4_K ~0.94 is the
fastest config). (Q8_0 here produced a longer utterance, so its wall is higher;
per-second-of-audio it is roughly on par with BF16.) One-time cost: in-memory
`quantize_onto` at load (~tens of seconds, CPU→Metal) — not counted in RTF since
it's a load-time cost, amortized across all syntheses in a session.

## Reproduce

```bash
# tiny-cpm (Q4_K talker, GPU sampling default; sub-realtime on an idle machine)
./target/release/tiny-cpm tts qwen3 models/Qwen3-TTS-12Hz-1.7B-Base \
  "你好，这是一段测试。很高兴见到你，今天天气真不错。" /tmp/t.wav --talker-quant q4_k

# audio.cpp (optimal Metal build, full sentence requires --reference-text)
cd ../audio.cpp
./build/macos-metal-opt/bin/audiocpp_cli --task tts --family qwen3_tts \
  --model $PWD/../tiny-cpm/models/Qwen3-TTS-12Hz-1.7B-Base --backend metal --log \
  --text "你好，这是一段测试。很高兴见到你，今天天气真不错。" \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature, others call me Mother Nature. I've been here for over four point five billion years, twenty-two thousand five hundred times longer than you." \
  --out /tmp/acpp.wav
```

## Gotcha that cost us a wrong conclusion

audio.cpp's Base model **truncates output at ~0.4 s if you don't pass `--reference-text`**
(it's documented in `webui/README.md`:356 — "voice cloning output is too short (ends at
~0.4s): ... `reference_text` is missing"). Our first audio.cpp runs silently produced
0.24–0.40 s stubs and looked "fast." Passing the reference transcript makes it emit the
full 6 s sentence and reveals the true RTF ~7.9. tiny-cpm's `--ref-text` has the same
contract (full ref transcript or the model babbles the ref) — see CLAUDE.md.
