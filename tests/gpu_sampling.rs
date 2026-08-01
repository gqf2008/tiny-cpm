//! Regression guard for the GPU-side codebook sampling used by the Qwen3-TTS code
//! predictor (`gpu_sample_token` in `talker.rs`, gated by `QWEN3_TTS_CPU_SAMPLE`).
//!
//! That path replaces 15 blocking per-step `to_vec2` readbacks with on-device
//! temperature + Gumbel-max argmax and a single end-of-frame readback. The semantics it
//! must preserve:
//!   1. **Greedy** (`do_sample=false`): the returned token is the argmax of the logits,
//!      matching the CPU `sample_from_logits_vec` reference exactly.
//!   2. **Gumbel-max** (`do_sample=true`): `argmax(logits + Gumbel(0,1))` is an exact
//!      softmax-weighted categorical draw — every sampled index lies in the vocab
//!      support, and over many draws the empirical distribution concentrates on the
//!      highest-probability token (never the suppressed tail).
//!
//! Runs on Metal. The tests exercise the REAL `gpu_sample_token` (exposed as
//! `#[doc(hidden)] pub` test-only from `talker.rs`); the local `gumbel_max` below
//! documents the algorithm and cross-checks that the real helper samples from the
//! same distribution. (An earlier version only re-implemented the math inline and
//! never touched the production helper — that gap is closed here.)

use candle_core::{Device, Tensor};

use tiny_cpm::models::qwen3_tts::talker::gpu_sample_token;

/// Gumbel-max categorical sample, identical math to `gpu_sample_token`'s sampled
/// path (kept as an independent algorithm reference for the cross-check below).
fn gumbel_max(logits: &Tensor) -> u32 {
    let u = Tensor::rand_like(logits, 1e-7, 1.0).unwrap();
    let g = u
        .log()
        .unwrap()
        .neg()
        .unwrap()
        .log()
        .unwrap()
        .neg()
        .unwrap();
    let perturbed = (logits + g).unwrap();
    perturbed
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()[0]
}

/// Run the REAL `gpu_sample_token` once and return the sampled index.
fn real_sample(
    logits: &Tensor,
    do_sample: bool,
    temperature: f64,
    top_k: usize,
    top_p: f32,
) -> u32 {
    gpu_sample_token(logits, do_sample, temperature, top_k, top_p, None)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()[0]
}

#[test]
fn gpu_sampling_semantics() {
    let dev = Device::new_metal(0).expect("metal device");

    // A peaked logits row: token 7 dominates; a suppressed tail below it.
    let vocab = 64usize;
    let mut v = vec![0.0f32; vocab];
    v[7] = 20.0; // dominant
    v[3] = 5.0;
    v[40] = -30.0; // heavily suppressed — should never win a softmax draw
    let logits = Tensor::from_vec(v.clone(), (1, vocab), &dev).unwrap();

    // (1) Greedy == CPU argmax (real function, do_sample=false).
    let cpu_argmax = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap();
    assert_eq!(cpu_argmax, 7);
    for _ in 0..5 {
        assert_eq!(
            real_sample(&logits, false, 0.9, 50, 1.0),
            cpu_argmax,
            "GPU greedy must match CPU argmax"
        );
    }

    // (2) Real Gumbel-max: 400 draws, all in support, dominant token wins the
    //     large majority, and the suppressed token is never drawn.
    let mut counts = [0u32; 64];
    let n = 400;
    for _ in 0..n {
        let t = real_sample(&logits, true, 0.9, 50, 1.0) as usize;
        assert!(t < vocab, "sampled index in support");
        counts[t] += 1;
    }
    assert_eq!(counts[40], 0, "suppressed tail never sampled");
    assert!(
        counts[7] as f32 / n as f32 > 0.9,
        "dominant token should win >90% of draws, got {}/{}",
        counts[7],
        n
    );

    // (3) Temperature sharpens: at temp -> 0 the draw collapses to argmax.
    for _ in 0..25 {
        assert_eq!(
            real_sample(&logits, true, 0.05, 50, 1.0),
            7,
            "low temperature collapses to argmax"
        );
    }

    // (4) Cross-check: the REAL helper and the inline algorithm reference sample
    //     from the same distribution (both concentrate on the dominant token).
    let mut inline_counts = [0u32; 64];
    for _ in 0..n {
        let t = gumbel_max(&logits) as usize;
        assert!(t < vocab, "inline sampled index in support");
        inline_counts[t] += 1;
    }
    assert_eq!(
        inline_counts[40], 0,
        "inline: suppressed tail never sampled"
    );
    assert!(
        inline_counts[7] as f32 / n as f32 > 0.9,
        "inline: dominant token should win >90% of draws, got {}/{}",
        inline_counts[7],
        n
    );

    println!(
        "gpu_sampling_semantics ok: argmax={}, real draws token7={}/{}, tail40={} | inline token7={}/{}",
        cpu_argmax, counts[7], n, counts[40], inline_counts[7], n
    );
}
