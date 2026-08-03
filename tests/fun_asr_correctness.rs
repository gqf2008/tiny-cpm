//! End-to-end correctness regression for Fun-ASR.
//!
//! Mirrors `tests/asr_correctness.rs` for the Fun-ASR-Nano LLM (Qwen3-0.6B, same
//! decoder dims as Qwen3-ASR). Fun-ASR SAMPLES by default, so the comparison
//! force-greeds (`ArgMax`) — deterministic, so the BF16 / Q8_0 / Q4_K greedy
//! transcripts must be token-exact against the committed golden. A token flip
//! means either the quant backbone wiring regressed OR the quant level genuinely
//! drifts on this clip.
//!
//! The golden was produced by the BF16 binary with `TINY_CPM_FUNASR_GREEDY=1` on
//! `models/Fun-ASR-Nano-2512/example/zh.mp3` and committed at
//! `tests/fixtures/fun_asr_zh.txt`.
//!
//! Skips (does NOT fail) when the model weights or audio are absent. Run with
//! weights present, in release:
//!   cargo test --release --test fun_asr_correctness -- --nocapture

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use tiny_cpm::exec::fun_asr_nano::FunAsrEngine;
use tiny_cpm::utils::audio_utils::load_audio_with_resample;

const MODEL_DIR: &str = "models/Fun-ASR-Nano-2512";
const AUDIO: &str = "models/Fun-ASR-Nano-2512/example/zh.mp3";
const GOLDEN: &str = "tests/fixtures/fun_asr_zh.txt";

/// Decode the fixed mp3 to 16kHz mono f32 (the 1-D `&[f32]` `transcribe_samples`
/// expects).
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

/// Shared body: load the engine (optionally quantized), force-greedy transcribe
/// the fixed audio, assert token-exact equality vs the BF16 golden.
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
    let mut engine = FunAsrEngine::load_with_quant(MODEL_DIR, &device, quant)
        .expect("load Fun-ASR engine");

    let samples = load_samples_16k_mono(AUDIO);
    let actual = engine
        .transcribe_samples(&samples, 512, true)
        .expect("transcribe_samples (force-greedy)");

    let golden = std::fs::read_to_string(GOLDEN)
        .expect("read golden")
        .trim()
        .to_string();

    eprintln!("[{label}] golden: {golden}");
    eprintln!("[{label}] actual: {actual}");
    assert_eq!(
        actual, golden,
        "[{label}] Fun-ASR greedy transcript drifted from BF16 golden — \
         a wiring/norm/RoPE/attention regression, or the quant level drifts on this clip"
    );
}

#[test]
fn funasr_transcript_matches_golden() {
    check_transcript_matches_golden(None, "bf16");
}

#[test]
fn funasr_q8_transcript_matches_golden() {
    check_transcript_matches_golden(Some(GgmlDType::Q8_0), "q8_0");
}

#[test]
fn funasr_q4k_transcript_matches_golden() {
    check_transcript_matches_golden(Some(GgmlDType::Q4K), "q4_k");
}
