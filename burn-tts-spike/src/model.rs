//! Shared tensor ops + Qwen3 decoder layer, ported from burn-asr-spike's
//! verified model.rs (which is itself a 1:1 port of tiny-cpm's candle code).
//! All the f16-range fixes carry over: rms_norm/layer_norm variance computed
//! in f32 (this model's bf16 activations also reach ~5.8e3, so x² overflows
//! f16's 65504 limit and would otherwise collapse every norm to zero).

use anyhow::Result;
use burn::tensor::{Bool, DType, Device, Int, Tensor, activation, ops::ConvOptions};

use burn::tensor::TensorData;
use std::collections::HashMap;

/// Talker dtype: f16 (cubecl-wgpu Metal has no BF16).
/// BURN_TTS_F32=1 overrides to f32 (diagnostic for f16 numeric issues).
pub fn dt() -> DType {
    if std::env::var("BURN_TTS_F32").is_ok() {
        DType::F32
    } else {
        DType::F16
    }
}
/// Codec dtype: F32, faithful to candle (the codec checkpoint is F32 and runs
/// F32 there; f16 is a possible later optimization).
pub const CODEC_DT: DType = DType::F32;

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

pub struct Weights {
    map: HashMap<String, TensorData>,
    device: Device,
}

impl Weights {
    pub fn new(map: HashMap<String, TensorData>, device: Device) -> Self {
        Self { map, device }
    }

    pub fn device(&self) -> Device {
        self.device.clone()
    }

    pub fn get<const D: usize>(&self, name: &str, dtype: DType) -> Result<Tensor<D>> {
        let td = self
            .map
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing weight: {name}"))?;
        Ok(Tensor::from_data(td.clone(), (&self.device, dtype)))
    }
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

pub fn linear(x: Tensor<3>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<3> {
    let w2 = w.swap_dims(0, 1).unsqueeze::<3>();
    let out = x.matmul(w2);
    match b {
        Some(b) => out + b.unsqueeze::<2>().unsqueeze::<3>(),
        None => out,
    }
}

/// candle RmsNorm: x * rsqrt(mean(x^2) + eps) * w. Variance in f32 (see header).
pub fn rms_norm(x: Tensor<3>, w: Tensor<1>, eps: f64) -> Tensor<3> {
    let dt = x.dtype();
    let xf = x.cast(DType::F32);
    let var = xf.clone().powf_scalar(2.0).mean_dim(2);
    let y = xf * (var + eps).powf_scalar(-0.5);
    y.cast(dt) * w.unsqueeze::<2>().unsqueeze::<3>()
}

/// rms_norm over the last dim of a rank-4 tensor (q/k norm). Variance in f32.
pub fn rms_norm4(x: Tensor<4>, w: Tensor<1>, eps: f64) -> Tensor<4> {
    let dt = x.dtype();
    let xf = x.cast(DType::F32);
    let var = xf.clone().powf_scalar(2.0).mean_dim(3);
    let y = xf * (var + eps).powf_scalar(-0.5);
    y.cast(dt) * w.unsqueeze::<2>().unsqueeze::<3>().unsqueeze::<4>()
}

/// candle LayerNorm (remove_mean=true, affine). Variance in f32.
pub fn layer_norm(x: Tensor<3>, w: Tensor<1>, b: Tensor<1>, eps: f64) -> Tensor<3> {
    let dt = x.dtype();
    let xf = x.cast(DType::F32);
    let centered = xf.clone() - xf.clone().mean_dim(2);
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    let y = centered * (var + eps).powf_scalar(-0.5);
    y.cast(dt) * w.unsqueeze::<2>().unsqueeze::<3>() + b.unsqueeze::<2>().unsqueeze::<3>()
}

/// Repeat each kv head n_rep times, candle layout: cat along the seq dim then
/// reshape so copies interleave ([h0,h0,h1,h1,...]) and q head i attends
/// kv head i/n_rep.
pub fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    let [b, h, s, dd] = x.dims();
    Tensor::cat(vec![x; n_rep], 2).reshape([b, h * n_rep, s, dd])
}

/// Eager attention: q (b,h,s,dd) × k/v (b,h,s_kv,dd) + mask + softmax.
pub fn eager_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    mask: Option<Tensor<4>>,
    scale: f64,
) -> Tensor<4> {
    let mut attn = q.matmul(k.swap_dims(2, 3)) * scale;
    if let Some(m) = mask {
        attn = attn + m;
    }
    let attn = activation::softmax(attn, 3);
    attn.matmul(v)
}

/// GPT-NeoX split-half rotation. cos/sin are (seq, dim) rank-2 (RoPE::forward
/// output), broadcast onto q (b, h, s, dim) as (1, 1, seq, dim).
pub fn apply_rope(x: Tensor<4>, cos: Tensor<2>, sin: Tensor<2>) -> Tensor<4> {
    let cos = cos.unsqueeze::<4>();
    let sin = sin.unsqueeze::<4>();
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rot = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos + rot * sin
}

/// Standard RoPE cos/sin: positions × inv_freq, cat(freqs, freqs), cos/sin.
/// Returns (seq, dim) rank-2 tensors cast to `dtype` (like candle RoPE::forward).
pub fn rotary(
    device: &Device,
    inv_freq: &[f32],
    seqlen_offset: usize,
    seq_len: usize,
    dtype: DType,
) -> (Tensor<2>, Tensor<2>) {
    let half = inv_freq.len();
    let pos = Tensor::arange(
        seqlen_offset as i64..(seqlen_offset + seq_len) as i64,
        device,
    )
    .float()
    .reshape([seq_len, 1]);
    let inv = Tensor::<1>::from_floats(inv_freq, device).reshape([1, half]);
    let freqs = pos.matmul(inv); // (seq_len, half)
    let emb: Tensor<2> = Tensor::cat(vec![freqs.clone(), freqs], 1).cast(dtype); // (seq_len, dim)
    let cos = emb.clone().cos();
    let sin = emb.sin();
    (cos, sin)
}

/// (b, 1, s, s) additive causal mask, -inf strictly above the diagonal.
/// burn's triu_mask/tril_mask naming is inverted vs torch: tril_mask([s,s], 0)
/// is true strictly above the diagonal, which is what a causal mask needs.
pub fn causal_mask(device: &Device, b: usize, s: usize) -> Tensor<4> {
    let upper = Tensor::<2, Bool>::tril_mask([s, s], 0, device);
    let mask = Tensor::<2, burn::tensor::Float>::full([s, s], 0.0_f32, device)
        .mask_fill(upper, f32::NEG_INFINITY);
    mask.unsqueeze::<3>()
        .unsqueeze::<4>()
        .repeat_dim(0, b)
        .cast(dt())
}

// ---------------------------------------------------------------------------
// Qwen3 decoder layer (shared by the 28-layer talker and 5-layer predictor)
// ---------------------------------------------------------------------------

pub struct DecoderLayer {
    q_w: Tensor<2>,
    k_w: Tensor<2>,
    v_w: Tensor<2>,
    o_w: Tensor<2>,
    q_norm_w: Tensor<1>,
    k_norm_w: Tensor<1>,
    gate_w: Tensor<2>,
    up_w: Tensor<2>,
    down_w: Tensor<2>,
    in_ln_w: Tensor<1>,
    post_ln_w: Tensor<1>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    kv_cache: Option<(Tensor<4>, Tensor<4>)>,
}

impl DecoderLayer {
    pub fn new(
        w: &Weights,
        prefix: &str,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            q_w: w.get(&format!("{prefix}.self_attn.q_proj.weight"), dt())?,
            k_w: w.get(&format!("{prefix}.self_attn.k_proj.weight"), dt())?,
            v_w: w.get(&format!("{prefix}.self_attn.v_proj.weight"), dt())?,
            o_w: w.get(&format!("{prefix}.self_attn.o_proj.weight"), dt())?,
            q_norm_w: w.get(&format!("{prefix}.self_attn.q_norm.weight"), dt())?,
            k_norm_w: w.get(&format!("{prefix}.self_attn.k_norm.weight"), dt())?,
            gate_w: w.get(&format!("{prefix}.mlp.gate_proj.weight"), dt())?,
            up_w: w.get(&format!("{prefix}.mlp.up_proj.weight"), dt())?,
            down_w: w.get(&format!("{prefix}.mlp.down_proj.weight"), dt())?,
            in_ln_w: w.get(&format!("{prefix}.input_layernorm.weight"), dt())?,
            post_ln_w: w.get(&format!("{prefix}.post_attention_layernorm.weight"), dt())?,
            n_heads,
            n_kv_heads,
            head_dim,
            eps,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: Tensor<3>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        mask: Option<&Tensor<4>>,
    ) -> Tensor<3> {
        let residual = xs.clone();
        let h = rms_norm(xs, self.in_ln_w.clone(), self.eps);
        let h = self.attn(h, cos, sin, mask);
        let h = h + residual;
        let residual = h.clone();
        let h = rms_norm(h, self.post_ln_w.clone(), self.eps);
        let gate = activation::silu(linear(h.clone(), self.gate_w.clone(), None));
        let up = linear(h, self.up_w.clone(), None);
        let h = linear(gate * up, self.down_w.clone(), None);
        h + residual
    }

    /// QKNormAttention with KV cache (cat along the seq dim).
    fn attn(
        &mut self,
        xs: Tensor<3>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        mask: Option<&Tensor<4>>,
    ) -> Tensor<3> {
        let [b, s, _] = xs.dims();
        let q = linear(xs.clone(), self.q_w.clone(), None)
            .reshape([b, s, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let q = rms_norm4(q, self.q_norm_w.clone(), self.eps);
        let q = apply_rope(q, cos.clone(), sin.clone());
        let k = linear(xs.clone(), self.k_w.clone(), None)
            .reshape([b, s, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = rms_norm4(k, self.k_norm_w.clone(), self.eps);
        let k = apply_rope(k, cos.clone(), sin.clone());
        let v = linear(xs, self.v_w.clone(), None)
            .reshape([b, s, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        let (k, v) = match &self.kv_cache {
            Some((pk, pv)) => (
                Tensor::cat(vec![pk.clone(), k], 2),
                Tensor::cat(vec![pv.clone(), v], 2),
            ),
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let groups = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(k, groups);
        let v = repeat_kv(v, groups);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let out = eager_attention(q, k, v, mask.cloned(), scale);
        let out = out
            .swap_dims(1, 2)
            .reshape([b, s, self.n_heads * self.head_dim]);
        linear(out, self.o_w.clone(), None)
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// inv_freq = 1 / base^(2i/dim) for i in 0..dim/2.
pub fn compute_default_rope_parameters(dim: usize, base: f32) -> Vec<f32> {
    (0..dim)
        .step_by(2)
        .map(|i| 1.0_f32 / base.powf(i as f32 / dim as f32))
        .collect()
}
