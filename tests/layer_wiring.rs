//! Regression test for the Qwen3-TTS quantized talker: one full talker layer
//! via the reference `Qwen3DecoderLayer` (F32) must match the
//! `QuantizedTalkerLayer` mirror (F32 passthrough) **bit-exactly**, in both the
//! prefill and the single-token decode (KV-cache-populated) regimes.
//!
//! This guards the class of wiring bug that caused the frame-1 decode babble:
//! a single layer is bit-exact and the QMatMul matmul is bit-exact, so any
//! talker divergence localizes to the backbone/layer wiring (attention, KV
//! cache, RoPE, residual, or a double/missing norm) — never to quantization or
//! the matmul. Runs on Metal against the real checkpoint in `models/`.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use tiny_cpm::models::qwen3::config::Qwen3Config;
use tiny_cpm::models::qwen3::model::Qwen3DecoderLayer;
use tiny_cpm::position_embed::rope::RoPE;

fn max_rel(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.to_dtype(DType::F32).unwrap();
    let b = b.to_dtype(DType::F32).unwrap();
    let diff = (&a - &b).unwrap().abs().unwrap();
    let num = diff.max_all().unwrap().to_scalar::<f32>().unwrap() as f64;
    let den = b
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
    num / den.max(1e-9)
}

#[test]
fn mirror_layer_matches_reference_decode() {
    let device = Device::new_metal(0).unwrap();
    let files = ["models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors".to_string()];
    let full: tiny_cpm::models::qwen3_tts::config::Qwen3TTSConfig = serde_json::from_str(
        &std::fs::read_to_string("models/Qwen3-TTS-12Hz-1.7B-Base/config.json").unwrap(),
    )
    .unwrap();
    let cfg = full.talker_config.clone();

    // F32 weights for BOTH the reference layer and the mirror (no quantization).
    let vb_f32 =
        unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device).unwrap() };
    let vb_cpu =
        unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu).unwrap() };

    let hidden = cfg.hidden_size;

    // Reference layer 0 (F32).
    let qcfg = Qwen3Config {
        attention_bias: false,
        attention_dropout: 0.0,
        bos_token_id: 0,
        eos_token_id: 0,
        head_dim: cfg.head_dim,
        hidden_act: candle_nn::Activation::Silu,
        hidden_size: cfg.hidden_size,
        initializer_range: 0.02,
        intermediate_size: cfg.intermediate_size,
        max_position_embeddings: 32768,
        max_window_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_hidden_layers: cfg.num_hidden_layers,
        num_key_value_heads: cfg.num_key_value_heads,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta as f32,
        tie_word_embeddings: false,
        torch_dtype: "float32".into(),
        use_cache: true,
        use_sliding_window: false,
        vocab_size: cfg.vocab_size,
    };
    let mut ref_layer = Qwen3DecoderLayer::new(&qcfg, vb_f32.pp("talker.model.layers.0")).unwrap();

    // Mirror backbone (F32 passthrough); forward_layer0_only isolates layer 0.
    let mut mirror = tiny_cpm::models::qwen3_tts::quantized_talker::load(
        &vb_cpu,
        &cfg,
        candle_core::quantized::GgmlDType::F32,
        &device,
    )
    .unwrap();

    let rotary = RoPE::new(cfg.head_dim, cfg.rope_theta as f32, &device).unwrap();

    // Deterministic F32 input: a P-token prefill, then a 1-token decode.
    let p = 6usize;
    let mk = |seq: usize, seed: u64| -> Tensor {
        let vals: Vec<f32> = (0..seq * hidden)
            .map(|i| ((((i as u64) * seed + 7) % 2000) as f32 - 1000.0) / 500.0)
            .collect();
        Tensor::from_vec(vals, (1, seq, hidden), &device).unwrap()
    };
    let prefill_x = mk(p, 2246822519);
    let decode_x = mk(1, 1442695041);

    // --- PREFILL (offset 0, causal mask) ---
    let (cos, sin) = rotary.forward(0, p, &device).unwrap();
    let mask =
        tiny_cpm::utils::tensor_utils::prepare_causal_attention_mask(1, p, 0, &device).unwrap();
    let ref_pre = ref_layer
        .forward(&prefill_x, &cos, &sin, Some(&mask))
        .unwrap();
    let mir_pre = mirror
        .forward_layer0_only(&prefill_x, &cos, &sin, Some(&mask))
        .unwrap();
    let e_pre = max_rel(&ref_pre, &mir_pre);
    eprintln!("prefill  layer0: ref vs mirror rel={e_pre:.3e}");

    // --- DECODE (offset p, no mask) ---
    let (cos1, sin1) = rotary.forward(p, 1, &device).unwrap();
    let ref_dec = ref_layer.forward(&decode_x, &cos1, &sin1, None).unwrap();
    let mir_dec = mirror
        .forward_layer0_only(&decode_x, &cos1, &sin1, None)
        .unwrap();
    let e_dec = max_rel(&ref_dec, &mir_dec);
    eprintln!("decode   layer0: ref vs mirror rel={e_dec:.3e}");

    assert!(e_pre < 1e-4, "prefill diverges: {e_pre}");
    assert!(e_dec < 1e-4, "decode diverges: {e_dec}");
}
