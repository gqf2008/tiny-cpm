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
//! Runs on Metal. Self-contained: it re-implements the same Gumbel-max math inline so it
//! guards the *algorithm* without depending on the private helper's name.

use candle_core::{Device, Tensor};

/// Gumbel-max categorical sample, identical math to the talker's GPU path.
fn gumbel_max(logits: &Tensor) -> u32 {
    let u = Tensor::rand_like(logits, 1e-7, 1.0).unwrap();
    let g = u.log().unwrap().neg().unwrap().log().unwrap().neg().unwrap();
    let perturbed = (logits + g).unwrap();
    perturbed
        .argmax(candle_core::D::Minus1)
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

    // (1) Greedy == CPU argmax.
    let gpu_argmax = logits
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()[0];
    let cpu_argmax = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap();
    assert_eq!(gpu_argmax, cpu_argmax, "GPU argmax must match CPU argmax");
    assert_eq!(gpu_argmax, 7);

    // (2) Gumbel-max: 400 draws, all in support, dominant token wins the large majority,
    //     and the suppressed token is never drawn.
    let mut counts = [0u32; 64];
    let n = 400;
    for _ in 0..n {
        let t = gumbel_max(&logits) as usize;
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
    let sharp = (&logits / 0.05).unwrap();
    for _ in 0..25 {
        assert_eq!(gumbel_max(&sharp), 7, "low temperature collapses to argmax");
    }

    println!(
        "gpu_sampling_semantics ok: argmax={}, draws token7={}/{}, tail40={}",
        gpu_argmax, counts[7], n, counts[40]
    );
}
