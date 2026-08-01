//! Quantized Qwen3 decoder layer — mirror of `model::DecoderLayer` with the 7
//! heavy 2-D projections as Q4_K GEMVs (custom cubecl kernel, see qmat.rs).
//! Norms/attention stay in the talker dtype; the Q4K output is cast back to
//! `dt()` after each GEMV. Mirror of candle's quantized_talker.rs (which keeps
//! activations F32 throughout — here the F32 stream is confined to the GEMV).

use anyhow::Result;
use burn::tensor::{DType, Float, Int, Tensor};

use crate::config::TalkerConfig;
use crate::model::{
    Weights, apply_rope, causal_mask, compute_default_rope_parameters, dt, eager_attention,
    repeat_kv, rms_norm, rms_norm4, rotary,
};
use crate::qmat::Q4KMatmul;

/// One decoder layer: RMSNorm → per-head-QK-norm GQA attention with KV cache →
/// residual → RMSNorm → SwiGLU MLP → residual, with q/k/v/o/gate/up/down as
/// Q4_K GEMVs.
pub struct QuantDecoderLayer {
    q_w: Q4KMatmul,
    k_w: Q4KMatmul,
    v_w: Q4KMatmul,
    o_w: Q4KMatmul,
    gate_w: Q4KMatmul,
    up_w: Q4KMatmul,
    down_w: Q4KMatmul,
    q_norm_w: Tensor<1>,
    k_norm_w: Tensor<1>,
    in_ln_w: Tensor<1>,
    post_ln_w: Tensor<1>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    kv_cache: Option<(Tensor<4>, Tensor<4>)>,
}

impl QuantDecoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w: &Weights,
        prefix: &str,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f64,
        anchor: &crate::qmat::RTAnchor,
    ) -> Result<Self> {
        let qmat = |name: &str| -> Result<Q4KMatmul> {
            let wt: Tensor<2, Float> = w.get(name, DType::F32)?;
            let (n, k) = (wt.dims()[0], wt.dims()[1]);
            let v: Vec<f32> = wt
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{name} read: {e}"))?;
            Q4KMatmul::new(&v, k, n, &anchor.0)
        };
        Ok(Self {
            q_w: qmat(&format!("{prefix}.self_attn.q_proj.weight"))?,
            k_w: qmat(&format!("{prefix}.self_attn.k_proj.weight"))?,
            v_w: qmat(&format!("{prefix}.self_attn.v_proj.weight"))?,
            o_w: qmat(&format!("{prefix}.self_attn.o_proj.weight"))?,
            gate_w: qmat(&format!("{prefix}.mlp.gate_proj.weight"))?,
            up_w: qmat(&format!("{prefix}.mlp.up_proj.weight"))?,
            down_w: qmat(&format!("{prefix}.mlp.down_proj.weight"))?,
            q_norm_w: w.get(&format!("{prefix}.self_attn.q_norm.weight"), dt())?,
            k_norm_w: w.get(&format!("{prefix}.self_attn.k_norm.weight"), dt())?,
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
    ) -> Result<Tensor<3>> {
        let residual = xs.clone();
        let h = rms_norm(xs, self.in_ln_w.clone(), self.eps);
        let h = self.attn(h, cos, sin, mask)?;
        let h = h + residual;
        let residual = h.clone();
        let h = rms_norm(h, self.post_ln_w.clone(), self.eps);
        let gate = burn::tensor::activation::silu(self.gate_w.forward(h.clone())?.cast(dt()));
        let up = self.up_w.forward(h)?.cast(dt());
        let h = self.down_w.forward(gate * up)?.cast(dt());
        Ok(h + residual)
    }

    /// QKNormAttention with KV cache — mirror of DecoderLayer::attn.
    fn attn(
        &mut self,
        xs: Tensor<3>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        mask: Option<&Tensor<4>>,
    ) -> Result<Tensor<3>> {
        let [b, s, _] = xs.dims();
        let q = self
            .q_w
            .forward(xs.clone())?
            .cast(dt())
            .reshape([b, s, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let q = rms_norm4(q, self.q_norm_w.clone(), self.eps);
        let q = apply_rope(q, cos.clone(), sin.clone());
        let k = self
            .k_w
            .forward(xs.clone())?
            .cast(dt())
            .reshape([b, s, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = rms_norm4(k, self.k_norm_w.clone(), self.eps);
        let k = apply_rope(k, cos.clone(), sin.clone());
        let v = self
            .v_w
            .forward(xs)?
            .cast(dt())
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
        Ok(self.o_w.forward(out)?.cast(dt()))
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Quantized talker backbone (28 layers), drop-in for `Vec<DecoderLayer>`.
pub struct QuantTalkerBackbone {
    pub layers: Vec<QuantDecoderLayer>,
}

/// Quantized code-predictor backbone (5 layers).
pub struct QuantPredictorBackbone {
    pub layers: Vec<QuantDecoderLayer>,
}

impl QuantPredictorBackbone {
    pub fn new(
        w: &Weights,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f64,
        anchor: &crate::qmat::RTAnchor,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(QuantDecoderLayer::new(
                w,
                &format!("talker.code_predictor.model.layers.{i}"),
                n_heads,
                n_kv_heads,
                head_dim,
                eps,
                anchor,
            )?);
        }
        Ok(Self { layers })
    }

    pub fn clear_kv_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_cache();
        }
    }
}

impl QuantTalkerBackbone {
    pub fn new(w: &Weights, cfg: &TalkerConfig, anchor: &crate::qmat::RTAnchor) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantDecoderLayer::new(
                w,
                &format!("talker.model.layers.{i}"),
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
                anchor,
            )?);
        }
        Ok(Self { layers })
    }

    pub fn clear_kv_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_cache();
        }
    }
}
