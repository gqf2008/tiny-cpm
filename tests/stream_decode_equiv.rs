//! Streaming-codec equivalence probe for Qwen3-TTS.
//!
//! The `--stream` path decodes each chunk as a SINGLE sliding-window
//! `CodecDecoder::decode` with a manual left-context trim, while the batch path uses
//! `chunked_decode(300, 25)`. This test answers two things on Metal with the real codec:
//!   1. Is `decode` bit-deterministic for identical input (run twice)?
//!   2. Does the sliding-window single-`decode`+trim reproduce `chunked_decode` on the
//!      same codes (i.e. is the streamed PCM sample-identical to the batch PCM)?
//! A large gap in (2) means the streaming windows mis-approximate the batch windows.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use tiny_cpm::models::qwen3_tts::codec::SpeechTokenizer;
use tiny_cpm::models::qwen3_tts::config::SpeechTokenizerConfig;

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    (&a.to_dtype(DType::F32).unwrap() - &b.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// Sliding-window streaming decode (mirrors exec `synthesize_pcm_streaming`'s flush):
/// for each chunk boundary, decode `[emitted-ctx .. end]` and keep only the new tail.
fn streaming_decode(
    dec: &tiny_cpm::models::qwen3_tts::codec::CodecDecoder,
    codes: &Tensor, // (1, 16, T)
    first: usize,
    chunk: usize,
    ctx: usize,
) -> Tensor {
    let t = codes.dim(2).unwrap(); // codes: (1, 16, T) — T is dim 2
    let frame_samples = dec.frame_samples();
    let mut out: Vec<Tensor> = Vec::new();
    let mut emitted = 0usize;
    while emitted < t {
        let threshold = if emitted == 0 { first } else { chunk };
        let end = (emitted + threshold).min(t);
        let win_start = emitted.saturating_sub(ctx);
        let context = emitted - win_start;
        let window = codes.narrow(2, win_start, end - win_start).unwrap();
        let wav = dec.decode(&window).unwrap(); // (1,1,(end-win_start)*1920)
        let drop = context * frame_samples;
        let tail = wav.narrow(2, drop, wav.dim(2).unwrap() - drop).unwrap();
        out.push(tail);
        emitted = end;
    }
    Tensor::cat(&out, 2).unwrap()
}

#[test]
fn streaming_matches_batch_decode() {
    let device = Device::new_metal(0).unwrap();
    let dir = "models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer";
    let cfg: SpeechTokenizerConfig =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/config.json")).unwrap())
            .unwrap();
    let files: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "safetensors").then(|| p.to_string_lossy().to_string())
        })
        .collect();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device).unwrap() };
    let tok = SpeechTokenizer::new(vb, &cfg, true).unwrap();
    let dec = &tok.decoder;

    // Fixed pseudo-random codes (deterministic, no RNG): 74 frames ≈ 5.9 s.
    let t = 74usize;
    let mut v = Vec::with_capacity(16 * t);
    for i in 0..(16 * t) as u32 {
        v.push((i.wrapping_mul(2654435761) >> 8) % 2048);
    }
    let codes = Tensor::from_vec(v, (1, 16, t), &device).unwrap();

    // (1) determinism: single full-window decode twice.
    let a1 = dec.decode(&codes).unwrap();
    let a2 = dec.decode(&codes).unwrap();
    let det = max_abs(&a1, &a2);
    eprintln!("decode determinism (full window, run twice): max|Δ| = {det}");

    // (2) batch chunked_decode(300,25) vs sliding-window streaming decode.
    let batch = dec.chunked_decode(&codes, 300, 25).unwrap();
    // Sweep the streaming chunk size: as it approaches the batch's 300-frame window the
    // gap must collapse monotonically to ~0 — that confirms the divergence is the
    // window-size receptive-field effect (an approximation that vanishes with chunk
    // size), not a logic bug in the trim. chunk=300 must be sample-identical.
    let mut prev = f32::MAX;
    for (first, chunk) in [(12usize, 25usize), (50, 50), (300, 300)] {
        let streamed = streaming_decode(dec, &codes, first, chunk, 25);
        let n = batch.dim(2).unwrap().min(streamed.dim(2).unwrap());
        let b = batch.narrow(2, 0, n).unwrap();
        let s = streamed.narrow(2, 0, n).unwrap();
        let gap = max_abs(&b, &s);
        eprintln!("batch(300,25) vs stream(first={first},chunk={chunk}): max|Δ| = {gap}");
        assert!(
            gap <= prev + 1e-6,
            "gap must shrink as the stream chunk grows: {gap} > {prev}"
        );
        prev = gap;
    }
    // The 300-frame streaming window == the batch window: sample-identical.
    assert!(
        prev < 1e-4,
        "chunk=300 stream must match batch chunked_decode bit-for-bit, got {prev}"
    );
    // decode is deterministic (same input twice -> identical output).
    assert!(det < 1e-4, "decode is non-deterministic on Metal: {det}");
}
