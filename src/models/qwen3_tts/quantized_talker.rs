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
use crate::models::qwen3_tts::rope_fused::apply_rope_fused_cache;
use crate::models::qwen3_tts::swiglu_fused::swiglu_fused;

/// One decoder layer, semantically identical to `Qwen3DecoderLayer` (RMSNorm →
/// per-head-QK-norm GQA attention with KV cache → residual → RMSNorm → SwiGLU
/// MLP → residual), but with the 7 projections as QMatMul.
struct QuantizedTalkerLayer {
    /// Fused q/k/v projection: the three F32 weight matrices are stacked along the
    /// output dim and quantized as ONE QTensor, so a single GEMV launch produces
    /// `[q | k | v]` (split after the matmul). Cuts 3 launches/layer → 1.
    qkv_proj: QMatMul,
    o_proj: QMatMul,
    /// Fused gate/up projection: stacked along the output dim into one QTensor; one
    /// GEMV produces `[gate | up]`, which swiglu_fused consumes after a split.
    gate_up_proj: QMatMul,
    down_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    /// Output-row widths of the q / k / v slices inside the fused qkv_proj result
    /// (num_heads*head_dim, num_kv_heads*head_dim, num_kv_heads*head_dim).
    q_rows: usize,
    kv_rows: usize,
    /// Output-row width of one of gate / up inside the fused gate_up_proj result
    /// (intermediate_size).
    mlp_rows: usize,
    /// Preallocated KV buffers, `(kv_heads, kv_cap, head_dim)` — the fused RoPE
    /// kernel appends k/v directly (no per-step `Tensor::cat`), and decode
    /// attention reads them as strided views (the fused SDPA takes explicit
    /// strides). Both talker (cap = max generated frames) and code-predictor
    /// (cap = 17) paths share this shape.
    kv_k: Tensor,
    kv_v: Tensor,
    kv_pos: usize,
}

impl QuantizedTalkerLayer {
    /// Mirror of `Qwen3DecoderLayer::forward` + `QKNormAttention::forward` +
    /// `GateUpDownMLP::forward`. QMatMul takes/returns F32 (Metal kernel
    /// constraint), so the whole stream stays F32 here.
    ///
    /// KV cache: k/v are appended by the fused RoPE kernel directly into the
    /// preallocated `(kv_heads, kv_cap, head_dim)` buffers at `kv_pos` (no
    /// per-step `Tensor::cat`); decode attention consumes the narrowed strided
    /// views via the fused SDPA (explicit strides). The masked prefill path
    /// falls to the eager attention, which `contiguous()`s the views (copy —
    /// once per frame, before the 15 decode steps). `kv_pos` is bumped by
    /// `seq_len` here; the multi-step prefill (code predictor, 2 tokens) passes
    /// the position in/out.
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_pos: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (b_sz, q_len, _) = xs.dims3()?;
        // Fused QKV: one GEMV over the stacked [q|k|v] weight, then split the output
        // rows. Per-head QK RMSNorm on (b, q_len, heads, head_dim) before transpose+RoPE.
        let qkv = self.qkv_proj.forward(&xs)?; // (b, q_len, q_rows + 2*kv_rows)
        let q = qkv.narrow(2, 0, self.q_rows)?;
        let k = qkv.narrow(2, self.q_rows, self.kv_rows)?;
        let v = qkv.narrow(2, self.q_rows + self.kv_rows, self.kv_rows)?;
        let q = q.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = k.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        // Fused RoPE + KV append: ONE Metal kernel does the rotation AND writes
        // k/v into the preallocated KV buffers at kv_pos (falls back to the
        // composite `apply_rotary_pos_emb` internally on non-Metal / unexpected
        // shapes, keeping the CPU test path working). Returns strided k/v views
        // over the appended range — the fused SDPA consumes them without a copy.
        let (q, k, v) =
            apply_rope_fused_cache(&q, &k, &v, cos, sin, &self.kv_k, &self.kv_v, kv_pos)?;
        self.kv_pos = kv_pos + q_len;
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
        // Fused gate/up: one GEMV over the stacked [gate|up] weight, split, then SwiGLU
        // (fuse silu(gate)*up into one Metal kernel when possible).
        let gate_up = self.gate_up_proj.forward(&xs)?; // (b, q_len, 2*mlp_rows)
        let gate = gate_up.narrow(2, 0, self.mlp_rows)?;
        let up = gate_up.narrow(2, self.mlp_rows, self.mlp_rows)?;
        let gated = match swiglu_fused(&gate, &up) {
            Some(f) => f?,
            None => (ops::silu(&gate)? * up)?,
        };
        let xs = self.down_proj.forward(&gated)?;
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
    /// `kv_pos` is the KV-cache write position — the SAME position for every layer
    /// (each layer has its own buffer; the caller bumps it by T after the stack).
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
        kv_pos: usize,
    ) -> Result<Tensor> {
        // QMatMul needs F32 activations on Metal.
        let mut xs = xs.to_dtype(DType::F32)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, cos, sin, attention_mask, kv_pos)?;
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
        kv_pos: usize,
    ) -> Result<Tensor> {
        let xs = xs.to_dtype(DType::F32)?;
        self.layers[0].forward(&xs, cos, sin, attention_mask, kv_pos)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_pos = 0;
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
        kv_pos: usize,
    ) -> Result<Vec<Tensor>> {
        let mut xs = xs.to_dtype(DType::F32)?;
        let mut outs = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, cos, sin, attention_mask, kv_pos)?;
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

/// Quantize SEVERAL 2-D weights (CPU), stacked along the output rows (dim 0) into ONE
/// QTensor → a single `QMatMul` whose forward produces the concatenated output. The
/// caller splits the rows back out. Fusing q/k/v (or gate/up) into one GEMV cuts the
/// per-layer kernel-launch count, which is the predictor's binding constraint at m=1.
/// Stacking happens in F32 *before* quantization, so each output row is quantized by
/// the same Q4_K super-block scheme the separate path would use (numerics equivalent,
/// not bit-identical, to the unfused weights — validated by the ASR round-trip).
fn qmat_fused(
    vb: &VarBuilder,
    names: &[&str],
    quant: GgmlDType,
    device: &Device,
) -> Result<QMatMul> {
    let mut mats = Vec::with_capacity(names.len());
    for name in names {
        let t = vb
            .get_unchecked(name)
            .with_context(|| format!("talker weight `{name}`"))?
            .to_dtype(DType::F32)?;
        mats.push(t);
    }
    let refs: Vec<&Tensor> = mats.iter().collect();
    let stacked = Tensor::cat(&refs, 0).context("stack fused projection weights")?;
    let qt = QTensor::quantize_onto(&stacked, quant, device)
        .with_context(|| format!("quantize fused {names:?} to {quant:?}"))?;
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

/// Build a quantized Qwen3 layer-stack backbone from a **CPU-mmaped**
/// safetensors VarBuilder. `prefix` is the tensor-name root holding
/// `layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `layers.{i}.mlp.{gate,up,down}_proj.weight`,
/// the per-head `q_norm`/`k_norm`, and the layer norms — `"talker.model"` for the
/// talker, `"code_predictor.model"` for the code predictor. `quant` is typically
/// `GgmlDType::Q4K`. `kv_cap` is the per-head KV-cache capacity (the max total
/// sequence the caller will ever run — talker: max generated frames, predictor:
/// 2 prefill + 15 decode = 17). The final norm is NOT loaded here (the caller
/// owns it).
#[allow(clippy::too_many_arguments)]
pub fn load_stack(
    vb: &VarBuilder,
    prefix: &str,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    rms_norm_eps: f64,
    quant: GgmlDType,
    kv_cap: usize,
    device: &Device,
) -> Result<QuantizedTalkerBackbone> {
    let eps = rms_norm_eps;
    let m = vb.pp(prefix);
    let mut layers = Vec::with_capacity(num_hidden_layers);
    let q_rows = num_attention_heads * head_dim;
    let kv_rows = num_key_value_heads * head_dim;
    for i in 0..num_hidden_layers {
        let l = m.pp(format!("layers.{i}"));
        let a = l.pp("self_attn");
        let f = l.pp("mlp");
        // Preallocated (kv_heads, kv_cap, head_dim) buffers; the fused RoPE kernel
        // appends into them (initial garbage is never read before it is written).
        let kv_k = Tensor::zeros((num_key_value_heads, kv_cap, head_dim), DType::F32, device)?;
        let kv_v = Tensor::zeros((num_key_value_heads, kv_cap, head_dim), DType::F32, device)?;
        layers.push(QuantizedTalkerLayer {
            qkv_proj: qmat_fused(
                &a,
                &["q_proj.weight", "k_proj.weight", "v_proj.weight"],
                quant,
                device,
            )?,
            o_proj: qmat(&a, "o_proj.weight", quant, device)?,
            gate_up_proj: qmat_fused(&f, &["gate_proj.weight", "up_proj.weight"], quant, device)?,
            down_proj: qmat(&f, "down_proj.weight", quant, device)?,
            q_norm: norm(&a, "q_norm.weight", eps, device)?,
            k_norm: norm(&a, "k_norm.weight", eps, device)?,
            input_layernorm: norm(&l, "input_layernorm.weight", eps, device)?,
            post_attention_layernorm: norm(&l, "post_attention_layernorm.weight", eps, device)?,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups: num_attention_heads / num_key_value_heads,
            head_dim,
            q_rows,
            kv_rows,
            mlp_rows: intermediate_size,
            kv_k,
            kv_v,
            kv_pos: 0,
        });
    }
    Ok(QuantizedTalkerBackbone { layers })
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
    // The final norm (`talker.model.norm.weight`) is deliberately NOT loaded
    // here: `Talker` already owns and applies it once in `forward_step`.
    // KV capacity: 4096 covers prompt + ICL ref + the max generated frames.
    load_stack(
        vb,
        "talker.model",
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.rms_norm_eps,
        quant,
        4096,
        device,
    )
}
