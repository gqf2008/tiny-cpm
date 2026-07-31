//! Quantized Qwen3-TTS talker backbone — QMatMul mirror of the BF16
//! `Qwen3DecoderLayer` stack (`crate::models::qwen3::model::Qwen3DecoderLayer`,
//! which stays untouched). Only the heavy 2-D matmuls (attn q/k/v/o, mlp
//! gate/up/down) run through `candle_core::quantized::QMatMul`; the per-head
//! Q/K RMSNorms, the layer RMSNorms, and the final norm stay full-precision.
//!
//! Two differences from the cosyvoice3 `quantized_lm.rs` pattern this copies:
//! 1. **Qwen3, not Qwen2**: no QKV bias, and a **per-head Q/K RMSNorm**
//!    (`q_norm`/`k_norm` over `head_dim`) applied to the `(b, q_len, heads,
//!    head_dim)` projections BEFORE transpose + RoPE — mirroring
//!    `common::modules::QKNormAttention::forward` exactly.
//! 2. **Runtime in-memory quantization, no GGUF**: weights come from the
//!    official `model.safetensors` (mmaped on CPU) and are quantized in place
//!    via `QTensor::quantize_onto(.., GgmlDType, Metal)` — the same mechanism
//!    as `chat`'s bf16-dir → Q8 path. Tensor names are the original
//!    safetensors paths (`talker.model.layers.{i}.self_attn.q_proj.weight`, …),
//!    NOT llama.cpp `blk.N.*` names.
//!
//! Activations are F32 through the QMatMul layers: candle 0.11's Metal
//! quantized-matmul kernels take an F32 activation buffer (F16/BF16 not
//! supported). The caller (`Talker::forward_step`) casts the backbone output
//! back to `talker_dtype` before the `codec_head`. `quantize_onto` requires a
//! CPU source, so the safetensors VarBuilder must be mmaped on `Device::Cpu`.

use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, RmsNorm, VarBuilder, ops};

use crate::common::modules::eager_attention_forward;
use crate::models::qwen3_tts::config::TalkerConfig;
use crate::models::qwen3_tts::rope_fused::apply_rope_fused;

/// One decoder layer, semantically identical to `Qwen3DecoderLayer` (RMSNorm →
/// per-head-QK-norm GQA attention with KV cache → residual → RMSNorm → SwiGLU
/// MLP → residual), but with the 7 projections as QMatMul.
struct QuantizedTalkerLayer {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QuantizedTalkerLayer {
    /// Mirror of `Qwen3DecoderLayer::forward` + `QKNormAttention::forward` +
    /// `GateUpDownMLP::forward`. QMatMul takes/returns F32 (Metal kernel
    /// constraint), so the whole stream stays F32 here.
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (b_sz, q_len, _) = xs.dims3()?;
        // Per-head QK RMSNorm on (b, q_len, heads, head_dim) before transpose+RoPE.
        let q = self
            .q_proj
            .forward(&xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k =
            self.k_proj
                .forward(&xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        // Fused RoPE: one Metal kernel instead of the ~12-op composite (falls back
        // to `apply_rotary_pos_emb` internally on non-Metal / unexpected shapes).
        let (q, k) = apply_rope_fused(&q, &k, cos, sin)?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[prev_k, &k], 2)?,
                Tensor::cat(&[prev_v, &v], 2)?,
            ),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let attn = eager_attention_forward(
            &q,
            &k,
            &v,
            Some(self.num_kv_groups),
            attention_mask,
            1.0 / (self.head_dim as f64).sqrt(),
        )?;
        let attn = attn.reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        let xs = (self.o_proj.forward(&attn)? + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let gate = ops::silu(&self.gate_proj.forward(&xs)?)?;
        let xs = self
            .down_proj
            .forward(&(gate * self.up_proj.forward(&xs)?)?)?;
        Ok((residual + xs)?)
    }
}

/// Quantized Qwen3 backbone, drop-in for the `Vec<Qwen3DecoderLayer>` the
/// talker drives. Activations are F32 (see header). The final norm is NOT held
/// here — it lives in `Talker` and is applied once in `Talker::forward_step`
/// (applying it in the backbone as well would double-norm the hidden state).
pub struct QuantizedTalkerBackbone {
    layers: Vec<QuantizedTalkerLayer>,
}

impl QuantizedTalkerBackbone {
    /// Run the layer stack over `xs` (1, T, hidden), F32 in/out.
    /// `cos`/`sin`/`attention_mask` are built by the caller (`Talker::forward_step`).
    ///
    /// NOTE: like the `Full` path (`Vec<Qwen3DecoderLayer>`), this returns the
    /// **raw** layer-stack output WITHOUT the final norm — `Talker::forward_step`
    /// applies `self.norm` (and `codec_head`) itself. Applying the norm here too
    /// would double-norm the hidden state (a second RMSNorm re-scales the
    /// already-normalized vector), which is exactly the frame-1 decode divergence.
    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // QMatMul needs F32 activations on Metal.
        let mut xs = xs.to_dtype(DType::F32)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, cos, sin, attention_mask)?;
        }
        Ok(xs)
    }

    /// Test-only: run ONLY layer 0 over `xs` (no final norm), keeping its KV
    /// cache across calls. Used by `tests/layer_wiring.rs` to compare the
    /// mirror against the reference `Qwen3DecoderLayer` on the identical input.
    #[doc(hidden)]
    pub fn forward_layer0_only(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let xs = xs.to_dtype(DType::F32)?;
        self.layers[0].forward(&xs, cos, sin, attention_mask)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }

    /// Diagnostic: like `forward`, but returns each layer's output (F32) so a
    /// caller can find the first layer that diverges from the BF16 reference.
    pub fn forward_trace(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut xs = xs.to_dtype(DType::F32)?;
        let mut outs = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, cos, sin, attention_mask)?;
            outs.push(xs.clone());
        }
        Ok(outs)
    }
}

/// Quantize one 2-D safetensors weight (CPU) to a `QMatMul` on `device`.
fn qmat(vb: &VarBuilder, name: &str, quant: GgmlDType, device: &Device) -> Result<QMatMul> {
    let t = vb
        .get_unchecked(name)
        .with_context(|| format!("talker weight `{name}`"))?;
    let qt = QTensor::quantize_onto(&t, quant, device)
        .with_context(|| format!("quantize `{name}` to {quant:?}"))?;
    Ok(QMatMul::from_qtensor(qt)?)
}

/// Load a small (norm) weight to an F32 `RmsNorm` on `device`.
fn norm(vb: &VarBuilder, name: &str, eps: f64, device: &Device) -> Result<RmsNorm> {
    let w = vb
        .get_unchecked(name)
        .with_context(|| format!("talker weight `{name}`"))?
        .to_dtype(DType::F32)?
        .to_device(device)?;
    Ok(RmsNorm::new(w, eps))
}

/// Build the quantized talker backbone from a **CPU-mmaped** safetensors
/// VarBuilder rooted at the repo root (tensor names are prefixed with
/// `talker.model.` here). `quant` is typically `GgmlDType::Q4K`.
pub fn load(
    vb: &VarBuilder,
    cfg: &TalkerConfig,
    quant: GgmlDType,
    device: &Device,
) -> Result<QuantizedTalkerBackbone> {
    let eps = cfg.rms_norm_eps;
    let m = vb.pp("talker.model");
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let l = m.pp(format!("layers.{i}"));
        let a = l.pp("self_attn");
        let f = l.pp("mlp");
        layers.push(QuantizedTalkerLayer {
            q_proj: qmat(&a, "q_proj.weight", quant, device)?,
            k_proj: qmat(&a, "k_proj.weight", quant, device)?,
            v_proj: qmat(&a, "v_proj.weight", quant, device)?,
            o_proj: qmat(&a, "o_proj.weight", quant, device)?,
            gate_proj: qmat(&f, "gate_proj.weight", quant, device)?,
            up_proj: qmat(&f, "up_proj.weight", quant, device)?,
            down_proj: qmat(&f, "down_proj.weight", quant, device)?,
            q_norm: norm(&a, "q_norm.weight", eps, device)?,
            k_norm: norm(&a, "k_norm.weight", eps, device)?,
            input_layernorm: norm(&l, "input_layernorm.weight", eps, device)?,
            post_attention_layernorm: norm(&l, "post_attention_layernorm.weight", eps, device)?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            kv_cache: None,
        });
    }
    // The final norm (`talker.model.norm.weight`) is deliberately NOT loaded
    // here: `Talker` already owns and applies it once in `forward_step`.
    Ok(QuantizedTalkerBackbone { layers })
}
