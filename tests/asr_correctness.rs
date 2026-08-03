//! End-to-end correctness regression for Qwen3-ASR.
//!
//! Guards exactly the class of bug the ASR levers (decoder quantization,
//! preallocated-KV append) could introduce — wiring, double / missing RMSNorm,
//! RoPE layout, attention-kernel numerics: the greedy transcript of a fixed audio
//! must match a committed golden **exactly**. Qwen3-ASR is greedy
//! (`generation_config.json` `do_sample: false` -> `ArgMax`) and the fused-SDPA
//! decode path is deterministic, so the transcript is bit-stable across runs; a
//! single token flip fails a test.
//!
//! The golden was produced by the current BF16 binary on
//! `models/Fun-ASR-Nano-2512/example/zh.mp3` and committed at
//! `tests/fixtures/qwen3_asr_zh.txt`. Regenerate it only after an intentional
//! model/format change (and confirm the diff is expected).
//!
//! Three configs are pinned token-exact against the SAME golden: BF16 (the
//! reference), Q8_0, and Q4_K. On this audio greedy `ArgMax` is bit-stable under
//! both quant levels — if a future change flips a token here, either the quant
//! backbone wiring regressed OR the quant level genuinely drifts on this clip
//! (then relax to a CER threshold + investigate).
//!
//! Skips (does NOT fail) when the model weights or audio are absent, so plain
//! `cargo test` on a machine without `models/` stays green. Run explicitly with
//! weights present, in release (Metal + the real decode path):
//!   cargo test --release --test asr_correctness -- --nocapture

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use tiny_cpm::exec::qwen3_asr::Qwen3AsrEngine;
use tiny_cpm::utils::audio_utils::load_audio_with_resample;

const MODEL_DIR: &str = "models/Qwen3-ASR-0.6B";
const AUDIO: &str = "models/Fun-ASR-Nano-2512/example/zh.mp3";
const GOLDEN: &str = "tests/fixtures/qwen3_asr_zh.txt";

/// Decode the fixed mp3 to 16kHz mono f32 (the 1-D `&[f32]` `transcribe_samples`
/// expects). `load_audio_with_resample` returns `(channels, samples)` for mono,
/// so squeeze the channel axis.
fn load_samples_16k_mono(path: &str) -> Vec<f32> {
    let audio = load_audio_with_resample(path, &Device::Cpu, Some(16000), Some(1))
        .expect("load audio");
    let audio = audio.to_dtype(DType::F32).expect("to f32");
    let flat = if audio.rank() == 2 {
        audio.squeeze(0).expect("squeeze channel")
    } else {
        audio
    };
    flat.to_vec1::<f32>().expect("to vec1")
}

/// Shared body: load the engine (optionally quantized), transcribe the fixed
/// audio, assert the cleaned transcript is token-exact vs the BF16 golden.
fn check_transcript_matches_golden(quant: Option<GgmlDType>, label: &str) {
    if !std::path::Path::new(MODEL_DIR).exists()
        || !std::path::Path::new(AUDIO).exists()
        || !std::path::Path::new(GOLDEN).exists()
    {
        eprintln!(
            "skip: {MODEL_DIR}, {AUDIO}, or {GOLDEN} absent (needs models/ weights + fixture)"
        );
        return;
    }

    let device = Device::new_metal(0).expect("Metal device");
    let mut engine = Qwen3AsrEngine::load_with_quant(MODEL_DIR, &device, quant)
        .expect("load Qwen3-ASR engine");

    let samples = load_samples_16k_mono(AUDIO);
    let actual = engine
        .transcribe_samples(&samples, 512)
        .expect("transcribe_samples");

    let golden = std::fs::read_to_string(GOLDEN)
        .expect("read golden")
        .trim()
        .to_string();

    eprintln!("[{label}] golden: {golden}");
    eprintln!("[{label}] actual: {actual}");
    assert_eq!(
        actual, golden,
        "[{label}] Qwen3-ASR greedy transcript drifted from BF16 golden — \
         a wiring/norm/RoPE/attention regression, or the quant level drifts on this clip"
    );
}

#[test]
fn qwen3_asr_transcript_matches_golden() {
    check_transcript_matches_golden(None, "bf16");
}

#[test]
fn qwen3_asr_q8_transcript_matches_golden() {
    check_transcript_matches_golden(Some(GgmlDType::Q8_0), "q8_0");
}

#[test]
fn qwen3_asr_q4k_transcript_matches_golden() {
    check_transcript_matches_golden(Some(GgmlDType::Q4K), "q4_k");
}
