//! Regression tests for the Qwen3-TTS quantized talker wiring.
//!
//! **Reference-vs-mirror tests**: one full talker layer (and one code-predictor
//! layer) via the reference `Qwen3DecoderLayer` (F32) must match the
//! `QuantizedTalkerLayer` mirror (F32 passthrough) to **≈1e-4 relative** (the
//! mirror's fused QKV/gate-up projections reorder the matmul math vs the
//! separate reference projections, so bit-exactness is not expected; `max_rel`
//! thresholds at 1e-4), in both the prefill and the single-token decode
//! (KV-cache-populated) regimes.
//!
//! This guards the class of wiring bug that caused the frame-1 decode babble:
//! a single layer matches to ~1e-4 and the QMatMul matmul is bit-exact, so any
//! talker divergence localizes to the backbone/layer wiring (attention, KV
//! cache, RoPE, residual, or a double/missing norm) — never to quantization or
//! the matmul. The predictor variant extends the same double-RMSNorm guard to
//! the 5-layer code predictor stack.
//!
//! **Fused-vs-composite tests**: the fused kernels (`rope_fused`, `sdpa_fused`,
//! and by extension the `qknorm_rope_fused` lineage in `QKNormAttention`) must
//! reproduce the eager composite math to ~1e-4 relative, so a kernel bug
//! (wrong index/layout) is caught without needing the full model.
//!
//! All tests run on Metal against the real checkpoint in `models/` (the
//! fused-vs-composite ones need no weights but still require a Metal device).

use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;

use tiny_cpm::common::modules::eager_attention_forward;
use tiny_cpm::models::qwen3::config::Qwen3Config;
use tiny_cpm::models::qwen3::model::Qwen3DecoderLayer;
use tiny_cpm::models::qwen3_tts::config::{CodePredictorConfig, TalkerConfig};
use tiny_cpm::models::qwen3_tts::quantized_talker;
use tiny_cpm::models::qwen3_tts::rope_fused::apply_rope_fused;
use tiny_cpm::models::qwen3_tts::sdpa_fused::sdpa_vector_attention;
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

/// Build the `Qwen3Config` a reference layer needs from a `TalkerConfig` (the
/// mirror is F32, so `torch_dtype` is set to float32).
fn qwen3_cfg_talker(cfg: &TalkerConfig) -> Qwen3Config {
    Qwen3Config {
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
    }
}

/// Same, from a `CodePredictorConfig` (predictor stack has its own hyper-params).
fn qwen3_cfg_predictor(cfg: &CodePredictorConfig) -> Qwen3Config {
    Qwen3Config {
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
    }
}

fn load_tts_config() -> tiny_cpm::models::qwen3_tts::config::Qwen3TTSConfig {
    serde_json::from_str(
        &std::fs::read_to_string("models/Qwen3-TTS-12Hz-1.7B-Base/config.json").unwrap(),
    )
    .unwrap()
}

/// Deterministic F32 input: `(1, seq, hidden)`, values in roughly [-2, 2].
fn mk_input(device: &Device, seq: usize, hidden: usize, seed: u64) -> Tensor {
    let vals: Vec<f32> = (0..seq * hidden)
        .map(|i| ((((i as u64) * seed + 7) % 2000) as f32 - 1000.0) / 500.0)
        .collect();
    Tensor::from_vec(vals, (1, seq, hidden), &device).unwrap()
}

/// Reference layer 0 vs mirror layer 0, prefill + decode. Shared by the talker
/// and code-predictor checks; `prefix` is the safetensors root of the layer
/// stack (`talker.model` / `talker.code_predictor.model`).
fn check_layer0_matches(
    device: &Device,
    label: &str,
    prefix: &str,
    qcfg: &Qwen3Config,
    hidden: usize,
    theta: f32,
    vb_f32: &VarBuilder,
    vb_cpu: &VarBuilder,
) {
    let mut ref_layer = Qwen3DecoderLayer::new(qcfg, vb_f32.pp(prefix).pp("layers.0")).unwrap();

    let mut mirror = quantized_talker::load_stack(
        vb_cpu,
        prefix,
        qcfg.num_hidden_layers,
        qcfg.num_attention_heads,
        qcfg.num_key_value_heads,
        qcfg.head_dim,
        qcfg.intermediate_size,
        qcfg.rms_norm_eps,
        candle_core::quantized::GgmlDType::F32,
        4096, // kv_cap: 4096 covers prefill(6) + decode in this test
        &device,
    )
    .unwrap();

    let rotary = RoPE::new(qcfg.head_dim, theta, &device).unwrap();

    let p = 6usize;
    let prefill_x = mk_input(&device, p, hidden, 2246822519);
    let decode_x = mk_input(&device, 1, hidden, 1442695041);

    // --- PREFILL (offset 0, causal mask) ---
    let (cos, sin) = rotary.forward(0, p, &device).unwrap();
    let mask =
        tiny_cpm::utils::tensor_utils::prepare_causal_attention_mask(1, p, 0, &device).unwrap();
    let ref_pre = ref_layer
        .forward(&prefill_x, &cos, &sin, Some(&mask))
        .unwrap();
    let mir_pre = mirror
        .forward_layer0_only(&prefill_x, &cos, &sin, Some(&mask), 0)
        .unwrap();
    let e_pre = max_rel(&ref_pre, &mir_pre);
    eprintln!("{label} prefill  layer0: ref vs mirror rel={e_pre:.3e}");

    // --- DECODE (offset p, no mask) ---
    let (cos1, sin1) = rotary.forward(p, 1, &device).unwrap();
    let ref_dec = ref_layer.forward(&decode_x, &cos1, &sin1, None).unwrap();
    let mir_dec = mirror
        .forward_layer0_only(&decode_x, &cos1, &sin1, None, p)
        .unwrap();
    let e_dec = max_rel(&ref_dec, &mir_dec);
    eprintln!("{label} decode   layer0: ref vs mirror rel={e_dec:.3e}");

    assert!(e_pre < 1e-4, "{label} prefill diverges: {e_pre}");
    assert!(e_dec < 1e-4, "{label} decode diverges: {e_dec}");
}

#[test]
fn mirror_talker_layer_matches_reference_decode() {
    let device = Device::new_metal(0).unwrap();
    let full = load_tts_config();
    let cfg = full.talker_config.clone();

    // F32 weights for BOTH the reference layer and the mirror (no quantization).
    let files = ["models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors".to_string()];
    let vb_f32 =
        unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device).unwrap() };
    let vb_cpu =
        unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu).unwrap() };

    check_layer0_matches(
        &device,
        "talker",
        "talker.model",
        &qwen3_cfg_talker(&cfg),
        cfg.hidden_size,
        cfg.rope_theta as f32,
        &vb_f32,
        &vb_cpu,
    );
}

#[test]
fn mirror_predictor_layer_matches_reference_decode() {
    // Same pattern as the talker layer-0 check, but for layer 0 of the 5-layer
    // code predictor (`talker.code_predictor.model`). The predictor shares the
    // double-RMSNorm trap (its `forward_hidden` applies `norm` once after the
    // backbone), so a missing/extra norm in the Quant predictor path would show
    // up here as an O(1) divergence from the reference layer.
    let device = Device::new_metal(0).unwrap();
    let full = load_tts_config();
    let cpc = full.talker_config.code_predictor_config.clone();

    let files = ["models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors".to_string()];
    let vb_f32 =
        unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device).unwrap() };
    let vb_cpu =
        unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu).unwrap() };

    check_layer0_matches(
        &device,
        "code-predictor",
        "talker.code_predictor.model",
        &qwen3_cfg_predictor(&cpc),
        cpc.hidden_size,
        cpc.rope_theta as f32,
        &vb_f32,
        &vb_cpu,
    );
}

// --- Fused-vs-composite equivalence guards ---------------------------------

/// Local reimplementation of the composite RoPE (`rotate_half` +
/// `broadcast_mul`/`add`), matching `apply_rotary_pos_emb_composite`'s math for
/// 4-D `(b, heads, seq, head_dim)` inputs with `(seq, head_dim)` cos/sin.

/// Deterministic F32 pseudo-random values in [-2, 2]. `Tensor::randn` creates an
/// F64 tensor, and candle's Metal backend has no F64 `rand_uniform`, so the fused
/// tests build inputs explicitly.
fn mk_rand(device: &Device, shape: (usize, usize, usize, usize), seed: u64) -> Tensor {
    let (a, b, c, d) = shape;
    let vals: Vec<f32> = (0..a * b * c * d)
        .map(|i| ((((i as u64).wrapping_mul(seed) + 7) % 2000) as f32 - 1000.0) / 500.0)
        .collect();
    Tensor::from_vec(vals, shape, device).unwrap()
}

fn composite_rotate_half(x: &Tensor) -> Tensor {
    let half_dim = x.dim(D::Minus1).unwrap() / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim).unwrap();
    let x2 = x
        .narrow(D::Minus1, half_dim, half_dim)
        .unwrap()
        .affine(-1.0, 0.0)
        .unwrap();
    Tensor::cat(&[&x2, &x1], D::Minus1)
        .unwrap()
        .contiguous()
        .unwrap()
}

fn composite_rope(q: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    // (seq, head_dim) -> (1, 1, seq, head_dim), then q*cos + rot(q)*sin.
    let cos = cos.unsqueeze(0).unwrap().unsqueeze(0).unwrap();
    let sin = sin.unsqueeze(0).unwrap().unsqueeze(0).unwrap();
    q.broadcast_mul(&cos)
        .unwrap()
        .add(&composite_rotate_half(q).broadcast_mul(&sin).unwrap())
        .unwrap()
}

/// `apply_rope_fused` must reproduce the composite RoPE math. This is the
/// regression guard for the fused-RoPE family (including the `qknorm_rope_fused`
/// kernel wired into `QKNormAttention`, which applies the same rotary math in a
/// single Metal pass).
#[test]
fn fused_rope_matches_composite() {
    let device = Device::new_metal(0).unwrap();
    let head_dim = 128usize;
    let seq = 5usize;
    let q_heads = 4usize;
    let kv_heads = 2usize;

    let q = mk_rand(&device, (1, q_heads, seq, head_dim), 42);
    let k = mk_rand(&device, (1, kv_heads, seq, head_dim), 43);
    let rotary = RoPE::new(head_dim, 1_000_000.0, &device).unwrap();
    let (cos, sin) = rotary.forward(0, seq, &device).unwrap(); // (seq, head_dim)

    let (q_f, k_f) = apply_rope_fused(&q, &k, &cos, &sin).unwrap();
    let e_q = max_rel(&q_f, &composite_rope(&q, &cos, &sin));
    let e_k = max_rel(&k_f, &composite_rope(&k, &cos, &sin));
    eprintln!("fused rope: q rel={e_q:.3e} k rel={e_k:.3e}");
    assert!(e_q < 1e-4, "fused RoPE q diverges from composite: {e_q}");
    assert!(e_k < 1e-4, "fused RoPE k diverges from composite: {e_k}");
}

/// `sdpa_vector_attention` (the fused flash-decode kernel) must reproduce the
/// eager matmul→scale→softmax→matmul attention math for a single-query
/// decode step. The eager reference runs through `eager_attention_forward` with
/// the qwen3-tts SDPA fast path opt-in OFF (the default), so it is the true
/// composite path.
#[test]
fn fused_sdpa_matches_eager() {
    let device = Device::new_metal(0).unwrap();
    let head_dim = 128usize;
    let q_heads = 4usize;
    let kv_heads = 2usize;
    let kv_seq = 5usize;

    // q (b, q_heads, 1, hd); k/v (b, kv_heads, kv_seq, hd) — the decode shape.
    let q = mk_rand(&device, (1, q_heads, 1, head_dim), 44);
    let k = mk_rand(&device, (1, kv_heads, kv_seq, head_dim), 45);
    let v = mk_rand(&device, (1, kv_heads, kv_seq, head_dim), 46);
    let scale = 1.0 / (head_dim as f64).sqrt();

    let fused = sdpa_vector_attention(&q, &k, &v, scale as f32)
        .unwrap()
        .expect("fused SDPA rejected the decode shape");
    let fused = fused.transpose(1, 2).unwrap().contiguous().unwrap(); // (b, 1, q_heads, hd)
    let eager = eager_attention_forward(&q, &k, &v, Some(q_heads / kv_heads), None, scale).unwrap();

    let e = max_rel(&fused, &eager);
    eprintln!("fused sdpa vs eager: rel={e:.3e}");
    assert!(e < 1e-4, "fused SDPA diverges from eager attention: {e}");
}
