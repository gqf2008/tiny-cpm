//! Qwen3-TTS 12 Hz speech tokenizer (codec), ported from
//! github.com/QwenLM/Qwen3-TTS qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py
//! (decoder + split-RVQ) and the transformers Mimi encoder it reuses
//! (transformers/models/mimi/modeling_mimi.py, v4.57.3).
//!
//! Two halves sharing one checkpoint (`speech_tokenizer/model.safetensors`, F32):
//! - [`CodecEncoder`]  (voice cloning only): 24 kHz wav → 16 RVQ codes/frame @ 12.5 Hz.
//!   SEANet conv stack (strides 4,5,6,8 → 25 Hz) → 8-layer transformer (LayerNorm+bias,
//!   sliding-window 250, LayerScale, GELU fc MLP, RoPE θ1e4) → replicate-pad downsample
//!   conv (→12.5 Hz) → split RVQ (1 semantic + 15 acoustic of the 31 loaded).
//! - [`CodecDecoder`]  (every synthesis): 16 codes/frame → 24 kHz wav. Split-RVQ dequant →
//!   pre_conv → pre-transformer (RMSNorm, sliding-window 72, gated SiLU, LayerScale, RoPE θ1e4)
//!   → 2× (ConvTranspose + ConvNeXt) → conv → 4× DecoderBlock (SnakeBeta + ConvTranspose +
//!   3 residual units) → SnakeBeta → conv → clamp(-1,1).
//!
//! Causal-conv padding is EnCodec-exact (`Qwen3TTSTokenizerV2CausalConvNet`): left pad
//! `kernel_effective - stride` zeros + a computed right "extra padding" so length stays
//! divisible; transposed convs trim `kernel - stride` samples from the right
//! (`Qwen3TTSTokenizerV2CausalTransConvNet`). All convs here are plain (no weight-norm).
use anyhow::{Context, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, Module,
    RmsNorm, VarBuilder, conv1d, conv1d_no_bias, layer_norm, linear, linear_no_bias, rms_norm,
};

use crate::models::qwen3_tts::config::{CodecDecoderConfig, CodecEncoderConfig};
use crate::position_embed::rope::{RoPE, apply_rotary_pos_emb};

// ---------------------------------------------------------------------------
// Causal conv helpers (EnCodec-exact padding).
// ---------------------------------------------------------------------------

/// `_get_extra_padding_for_conv1d`: right zero-pad so `(len - k_eff + pad) / stride + 1`
/// is integral, where `pad = k_eff - stride` (the left pad).
fn conv1d_extra_padding(len: usize, kernel_eff: usize, padding: usize, stride: usize) -> usize {
    let n_frames = (len as f64 - kernel_eff as f64 + padding as f64) / stride as f64 + 1.0;
    let ideal = (n_frames.ceil() as usize - 1) * stride + (kernel_eff - padding);
    ideal.saturating_sub(len)
}

/// Left zero-pad `left` + right zero-pad `right` on the last dim of a (B, C, T) tensor.
fn pad_zeros_last_dim(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let (b, c, _t) = x.dims3()?;
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        parts.push(Tensor::zeros((b, c, left), x.dtype(), x.device())?);
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(Tensor::zeros((b, c, right), x.dtype(), x.device())?);
    }
    Ok(Tensor::cat(&parts, D::Minus1)?)
}

/// `Qwen3TTSTokenizerV2CausalConvNet`: left-pad (kernel_eff-stride) zeros + right extra-pad,
/// then a plain conv1d (no built-in padding).
#[derive(Debug, Clone)]
struct CausalConv {
    conv: Conv1d,
    stride: usize,
    kernel_eff: usize, // (kernel-1)*dilation + 1
    padding: usize,    // kernel_eff - stride (left pad)
}

impl CausalConv {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        // The checkpoint nests every conv's tensors under `.conv` (`....conv.weight/bias`).
        let vb = vb.pp("conv");
        let conv = if bias {
            conv1d(in_c, out_c, k, cfg, vb)?
        } else {
            conv1d_no_bias(in_c, out_c, k, cfg, vb)?
        };
        let kernel_eff = (k - 1) * dilation + 1;
        Ok(Self {
            conv,
            stride,
            kernel_eff,
            padding: kernel_eff - stride,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let len = x.dim(D::Minus1)?;
        let extra = conv1d_extra_padding(len, self.kernel_eff, self.padding, self.stride);
        let x = pad_zeros_last_dim(x, self.padding, extra)?;
        Ok(self.conv.forward(&x)?)
    }
}

/// `Qwen3TTSTokenizerV2CausalTransConvNet`: plain conv_transpose1d then trim `kernel-stride`
/// from the right.
#[derive(Debug, Clone)]
struct CausalTransConv {
    conv: ConvTranspose1d,
    right_pad: usize,
}

impl CausalTransConv {
    fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let w = vb.get((in_c, out_c, k), "conv.weight")?;
        let conv = if bias {
            let b = vb.get(out_c, "conv.bias")?;
            ConvTranspose1d::new(w, Some(b), cfg)
        } else {
            ConvTranspose1d::new(w, None, cfg)
        };
        Ok(Self {
            conv,
            right_pad: k.saturating_sub(stride),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        if self.right_pad > 0 {
            let t = y.dim(D::Minus1)?;
            Ok(y.narrow(D::Minus1, 0, t - self.right_pad)?)
        } else {
            Ok(y)
        }
    }
}

// ---------------------------------------------------------------------------
// SnakeBeta activation.
// ---------------------------------------------------------------------------

/// `SnakeBeta`: x + (1/(exp(beta)+1e-9)) * sin^2(x * exp(alpha)), per-channel alpha/beta.
#[derive(Debug, Clone)]
struct SnakeBeta {
    alpha: Tensor, // (C,)
    beta: Tensor,  // (C,)
}

impl SnakeBeta {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let alpha = vb.get(channels, "alpha")?;
        let beta = vb.get(channels, "beta")?;
        Ok(Self { alpha, beta })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, c, _t) = x.dims3()?;
        let alpha = self.alpha.exp()?.reshape((1, c, 1))?;
        let beta = self.beta.exp()?.reshape((1, c, 1))?;
        let periodic = x
            .broadcast_mul(&alpha)?
            .sin()?
            .sqr()?
            .broadcast_div(&(beta + 1e-9)?)?;
        Ok((x + periodic)?)
    }
}

// ---------------------------------------------------------------------------
// Decoder RVQ (dequant).
// ---------------------------------------------------------------------------

/// One EMA Euclidean codebook, decode-side only: `embedding = embedding_sum / clamp(cluster_usage, 1e-5)`.
#[derive(Debug, Clone)]
struct EuclideanCodebookDecode {
    embedding: Tensor, // (codebook_size, dim) precomputed
}

impl EuclideanCodebookDecode {
    fn new(vb: VarBuilder) -> Result<Self> {
        // NOTE: decoder files spell it `embedding_sum` (encoder uses `embed_sum`).
        let embedding_sum = vb
            .get((2048, 256), "embedding_sum")
            .or_else(|_| vb.get((2048, 256), "embed_sum"))
            .context("codebook embedding_sum/embed_sum")?;
        let cluster_usage = vb.get(2048, "cluster_usage")?;
        let embedding = embedding_sum.to_dtype(DType::F32)?.broadcast_div(
            &cluster_usage
                .to_dtype(DType::F32)?
                .clamp(1e-5, f64::INFINITY)?
                .unsqueeze(1)?,
        )?;
        Ok(Self { embedding })
    }
    /// codes: (B, T) → (B, T, dim).
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let b = codes.dim(0)?;
        let t = codes.dim(1)?;
        let flat = codes.flatten_all()?.to_dtype(DType::U32)?;
        Ok(self
            .embedding
            .index_select(&flat, 0)?
            .reshape((b, t, self.embedding.dim(1)?))?)
    }
}

/// `VectorQuantization.decode`: codebook decode → transpose to (B, dim, T).
/// (project_out is Identity here — codebook_dim 256 == dim 256 — so omitted.)
struct VectorQuantizationDecode {
    codebook: EuclideanCodebookDecode,
}

impl VectorQuantizationDecode {
    /// codes (B, T) → (B, dim, T)
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let q = self.codebook.decode(codes)?; // (B, T, dim)
        Ok(q.transpose(1, 2)?) // (B, dim, T)
    }
}

/// `ResidualVectorQuantizer.decode`: sum per-layer dequant + 1×1 output_proj conv.
struct RvqDecode {
    layers: Vec<VectorQuantizationDecode>,
    output_proj: Conv1d, // 1×1, dim 256 → 512
}

impl RvqDecode {
    fn new(vb: VarBuilder, n_layers: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(VectorQuantizationDecode {
                codebook: EuclideanCodebookDecode::new(
                    vb.pp("vq").pp("layers").pp(i).pp("_codebook"),
                )?,
            });
        }
        let output_proj =
            conv1d_no_bias(256, 512, 1, Conv1dConfig::default(), vb.pp("output_proj"))?;
        Ok(Self {
            layers,
            output_proj,
        })
    }
    /// codes: slice of per-layer (B, T) → (B, 512, T)
    fn decode(&self, codes: &[Tensor]) -> Result<Tensor> {
        anyhow::ensure!(codes.len() == self.layers.len(), "rvq layer count mismatch");
        let mut acc: Option<Tensor> = None;
        for (layer, codes_i) in self.layers.iter().zip(codes.iter()) {
            let q = layer.decode(codes_i)?; // (B, dim, T)
            acc = Some(match acc {
                None => q,
                Some(a) => (a + q)?,
            });
        }
        let q = acc.ok_or_else(|| anyhow!("rvq: no layers"))?;
        Ok(self.output_proj.forward(&q)?)
    }
}

/// `SplitResidualVectorQuantizer.decode` (decoder side): rvq_first (1 semantic) + rvq_rest (15 acoustic).
struct SplitRvqDecode {
    rvq_first: RvqDecode,
    rvq_rest: RvqDecode,
}

impl SplitRvqDecode {
    fn new(vb: VarBuilder) -> Result<Self> {
        let rvq_first = RvqDecode::new(vb.pp("rvq_first"), 1)?;
        let rvq_rest = RvqDecode::new(vb.pp("rvq_rest"), 15)?;
        Ok(Self {
            rvq_first,
            rvq_rest,
        })
    }
    /// codes (B, 16, T) → (B, 512, T)
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let b = codes.dim(0)?;
        let t = codes.dim(2)?;
        let first = vec![codes.narrow(1, 0, 1)?.reshape((b, t))?];
        let mut rest = Vec::with_capacity(15);
        for i in 1..16 {
            rest.push(codes.narrow(1, i, 1)?.reshape((b, t))?);
        }
        let q = self.rvq_first.decode(&first)?;
        let q2 = self.rvq_rest.decode(&rest)?;
        Ok((q + q2)?)
    }
}

// ---------------------------------------------------------------------------
// Decoder pre-transformer (RMSNorm, no qk-norm, sliding-window 72, gated SiLU, LayerScale).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LayerScale {
    scale: Tensor,
}
impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            scale: vb.get(dim, "scale")?,
        })
    }
    /// x: (B, T, C) — scale broadcast over the last dim.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_mul(&self.scale)?)
    }
}

/// Sliding-window causal mask: query i attends keys [i-window+1, i]. Additive (1,1,T,T) f32.
fn sliding_window_causal_mask(t: usize, window: usize, device: &Device) -> Result<Tensor> {
    let mut m = vec![f32::NEG_INFINITY; t * t];
    for i in 0..t {
        let lo = i.saturating_sub(window - 1);
        for (j, item) in m.iter_mut().enumerate().skip(i * t).take(t) {
            if j - i * t >= lo && j - i * t <= i {
                *item = 0.0;
            }
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, t, t), device)?)
}

struct DecoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
    scaling: f64,
}

impl DecoderAttention {
    fn new(vb: VarBuilder, cfg: &CodecDecoderConfig) -> Result<Self> {
        let nh = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let kvh = cfg.num_key_value_heads;
        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, nh * hd, vb.pp("self_attn.q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kvh * hd, vb.pp("self_attn.k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kvh * hd, vb.pp("self_attn.v_proj"))?,
            o_proj: linear_no_bias(nh * hd, cfg.hidden_size, vb.pp("self_attn.o_proj"))?,
            num_heads: nh,
            head_dim: hd,
            num_kv_groups: nh / kvh,
            scaling: (hd as f64).powf(-0.5),
        })
    }

    /// x: (B, T, hidden). Full attention with sliding-window causal mask (no KV cache).
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let kv_heads = self.num_heads / self.num_kv_groups;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        // q/k norm are Identity in this decoder; apply RoPE (θ1e4).
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin, true)?;
        let k = crate::utils::tensor_utils::repeat_kv(k, self.num_kv_groups)?;
        let v = crate::utils::tensor_utils::repeat_kv(v, self.num_kv_groups)?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn = (q.matmul(&k.t()?.contiguous()?)? * self.scaling)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.contiguous()?;
        let out = attn.matmul(&v)?; // (B, H, T, hd)
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, t, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }
}

struct DecoderTransformerLayer {
    attn: DecoderAttention,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn_layer_scale: LayerScale,
    mlp_layer_scale: LayerScale,
}

impl DecoderTransformerLayer {
    fn new(vb: VarBuilder, cfg: &CodecDecoderConfig) -> Result<Self> {
        Ok(Self {
            attn: DecoderAttention::new(vb.clone(), cfg)?,
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.gate_proj"),
            )?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))?,
            down_proj: linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("mlp.down_proj"),
            )?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            self_attn_layer_scale: LayerScale::new(
                vb.pp("self_attn_layer_scale"),
                cfg.hidden_size,
            )?,
            mlp_layer_scale: LayerScale::new(vb.pp("mlp_layer_scale"), cfg.hidden_size)?,
        })
    }
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h, cos, sin, mask)?;
        let h = (residual + self.self_attn_layer_scale.forward(&h)?)?;
        let residual = h.clone();
        let m = self.post_attention_layernorm.forward(&h)?;
        let m = self
            .down_proj
            .forward(&(m.apply(&self.gate_proj)?.silu()? * m.apply(&self.up_proj)?)?)?;
        Ok((residual + self.mlp_layer_scale.forward(&m)?)?)
    }
}

struct DecoderPreTransformer {
    input_proj: Linear,
    output_proj: Linear,
    layers: Vec<DecoderTransformerLayer>,
    norm: RmsNorm,
    rotary: RoPE,
    sliding_window: usize,
}

impl DecoderPreTransformer {
    fn new(vb: VarBuilder, cfg: &CodecDecoderConfig) -> Result<Self> {
        let input_proj = linear(cfg.latent_dim, cfg.hidden_size, vb.pp("input_proj"))?;
        let output_proj = linear(cfg.hidden_size, cfg.latent_dim, vb.pp("output_proj"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderTransformerLayer::new(vb.pp("layers").pp(i), cfg)?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary = RoPE::new(cfg.head_dim, cfg.rope_theta as f32, vb.device())?;
        Ok(Self {
            input_proj,
            output_proj,
            layers,
            norm,
            rotary,
            sliding_window: cfg.sliding_window,
        })
    }
    /// x: (B, T, latent_dim) → (B, T, latent_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.input_proj.forward(x)?;
        let t = h.dim(1)?;
        let (cos, sin) = self.rotary.forward(0, t, h.device())?;
        let mask = sliding_window_causal_mask(t, self.sliding_window, h.device())?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask)?;
        }
        h = self.norm.forward(&h)?;
        Ok(self.output_proj.forward(&h)?)
    }
}

// ---------------------------------------------------------------------------
// ConvNeXt block (upsample) + waveform decoder blocks.
// ---------------------------------------------------------------------------

struct ConvNeXtBlock {
    dwconv: CausalConv, // depthwise k7
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let dwconv = CausalConv::new(vb.pp("dwconv"), dim, dim, 7, 1, 1, dim, true)?;
        let norm = layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }
    /// x: (B, C, T)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = x.clone();
        let h = self.dwconv.forward(x)?; // (B,C,T)
        let h = h.transpose(1, 2)?; // (B,T,C)
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?.gelu()?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.broadcast_mul(&self.gamma)?;
        let h = h.transpose(1, 2)?; // (B,C,T)
        Ok((input + h)?)
    }
}

struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv, // k7, dilation d
    act2: SnakeBeta,
    conv2: CausalConv, // k1
}

impl DecoderResidualUnit {
    fn new(vb: VarBuilder, dim: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(vb.pp("act1"), dim)?,
            conv1: CausalConv::new(vb.pp("conv1"), dim, dim, 7, 1, dilation, 1, true)?,
            act2: SnakeBeta::new(vb.pp("act2"), dim)?,
            conv2: CausalConv::new(vb.pp("conv2"), dim, dim, 1, 1, 1, 1, true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.act1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        Ok((h + residual)?)
    }
}

/// One `Qwen3TTSTokenizerV2DecoderDecoderBlock`: SnakeBeta → ConvTranspose → 3 residual units (dil 1,3,9).
struct DecoderBlock {
    snake: SnakeBeta,
    trans_conv: CausalTransConv,
    residuals: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    fn new(vb: VarBuilder, cfg: &CodecDecoderConfig, layer_idx: usize) -> Result<Self> {
        let in_dim = cfg.decoder_dim / 2usize.pow(layer_idx as u32);
        let out_dim = cfg.decoder_dim / 2usize.pow(layer_idx as u32 + 1);
        let rate = cfg.upsample_rates[layer_idx];
        let snake = SnakeBeta::new(vb.pp("block").pp(0), in_dim)?;
        // transpose-conv weight is stored [in, out, k] as `block.1.conv.weight`.
        let trans_conv =
            CausalTransConv::new(vb.pp("block").pp(1), in_dim, out_dim, 2 * rate, rate, true)?;
        let mut residuals = Vec::with_capacity(3);
        for (ri, dilation) in [1usize, 3, 9].iter().enumerate() {
            residuals.push(DecoderResidualUnit::new(
                vb.pp("block").pp(ri + 2),
                out_dim,
                *dilation,
            )?);
        }
        Ok(Self {
            snake,
            trans_conv,
            residuals,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.snake.forward(x)?;
        h = self.trans_conv.forward(&h)?;
        for r in &self.residuals {
            h = r.forward(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// CodecDecoder (public).
// ---------------------------------------------------------------------------

pub struct CodecDecoder {
    quantizer: SplitRvqDecode,
    pre_conv: CausalConv,
    pre_transformer: DecoderPreTransformer,
    upsample: Vec<(CausalTransConv, ConvNeXtBlock)>,
    decoder0: CausalConv,      // latent → decoder_dim, k7
    blocks: Vec<DecoderBlock>, // 4
    final_snake: SnakeBeta,
    final_conv: CausalConv, // out_dim → 1, k7
    total_upsample: usize,
}

impl CodecDecoder {
    pub fn new(vb: VarBuilder, cfg: &CodecDecoderConfig) -> Result<Self> {
        let quantizer = SplitRvqDecode::new(vb.pp("quantizer"))?;
        let pre_conv = CausalConv::new(
            vb.pp("pre_conv"),
            cfg.codebook_dim,
            cfg.latent_dim,
            3,
            1,
            1,
            1,
            true,
        )?;
        let pre_transformer = DecoderPreTransformer::new(vb.pp("pre_transformer"), cfg)?;
        let mut upsample = Vec::with_capacity(cfg.upsampling_ratios.len());
        for (i, &factor) in cfg.upsampling_ratios.iter().enumerate() {
            let tc = CausalTransConv::new(
                vb.pp("upsample").pp(i).pp(0),
                cfg.latent_dim,
                cfg.latent_dim,
                factor,
                factor,
                true,
            )?;
            let cn = ConvNeXtBlock::new(vb.pp("upsample").pp(i).pp(1), cfg.latent_dim)?;
            upsample.push((tc, cn));
        }
        let decoder0 = CausalConv::new(
            vb.pp("decoder").pp(0),
            cfg.latent_dim,
            cfg.decoder_dim,
            7,
            1,
            1,
            1,
            true,
        )?;
        let mut blocks = Vec::with_capacity(cfg.upsample_rates.len());
        for i in 0..cfg.upsample_rates.len() {
            blocks.push(DecoderBlock::new(vb.pp("decoder").pp(i + 1), cfg, i)?);
        }
        let n = cfg.upsample_rates.len();
        let out_dim = cfg.decoder_dim / 2usize.pow(n as u32);
        let final_snake = SnakeBeta::new(vb.pp("decoder").pp(n + 1), out_dim)?;
        let final_conv = CausalConv::new(vb.pp("decoder").pp(n + 2), out_dim, 1, 7, 1, 1, 1, true)?;
        let total_upsample = cfg.total_upsample();
        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            decoder0,
            blocks,
            final_snake,
            final_conv,
            total_upsample,
        })
    }

    /// codes: (B, 16, T) u32/i64 → waveform (B, 1, T*1920) f32 in [-1, 1].
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let hidden = self.quantizer.decode(codes)?; // (B, 512, T)
        let hidden = self.pre_conv.forward(&hidden)?.transpose(1, 2)?; // (B, T, latent)
        let hidden = self.pre_transformer.forward(&hidden)?; // (B, T, latent)
        let mut hidden = hidden.transpose(1, 2)?; // (B, latent, T)
        for (tc, cn) in &self.upsample {
            hidden = tc.forward(&hidden)?;
            hidden = cn.forward(&hidden)?;
        }
        let mut wav = self.decoder0.forward(&hidden)?;
        for block in &self.blocks {
            wav = block.forward(&wav)?;
        }
        wav = self.final_snake.forward(&wav)?;
        wav = self.final_conv.forward(&wav)?;
        Ok(wav.clamp(-1.0, 1.0)?)
    }

    /// Output PCM samples per codec frame (`total_upsample`; 1920 at 24 kHz).
    /// Used by the streaming path to slice a re-decoded window down to its new tail.
    pub fn frame_samples(&self) -> usize {
        self.total_upsample
    }

    /// `chunked_decode`: 300-frame chunks with 25-frame left context (context output trimmed).
    /// codes: (B, 16, T) → (B, 1, T*1920).
    pub fn chunked_decode(
        &self,
        codes: &Tensor,
        chunk_size: usize,
        left_context: usize,
    ) -> Result<Tensor> {
        let t_total = codes.dim(D::Minus1)?;
        let mut wavs = Vec::new();
        let mut start = 0usize;
        while start < t_total {
            let end = (start + chunk_size).min(t_total);
            let context = if start > left_context {
                left_context
            } else {
                start
            };
            let chunk = codes.narrow(D::Minus1, start - context, (end - start) + context)?;
            let wav = self.decode(&chunk)?;
            let drop = context * self.total_upsample;
            let wav = wav.narrow(D::Minus1, drop, wav.dim(D::Minus1)? - drop)?;
            wavs.push(wav);
            start = end;
        }
        Ok(Tensor::cat(&wavs, D::Minus1)?)
    }
}

// ---------------------------------------------------------------------------
// CodecEncoder (voice cloning): Mimi SEANet + transformer + split-RVQ encode.
// ---------------------------------------------------------------------------

/// Mimi residual block: ELU → conv k3 (dim→dim/2) → ELU → conv k1 (dim/2→dim), residual add.
struct MimiResnetBlock {
    conv1: CausalConv,
    conv2: CausalConv,
}

impl MimiResnetBlock {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        // vb points at `...block`; convs are at block.1 and block.3 (block.0/2 are ELU, no weights).
        let conv1 = CausalConv::new(vb.pp(1), dim, dim / 2, 3, 1, 1, 1, true)?;
        let conv2 = CausalConv::new(vb.pp(3), dim / 2, dim, 1, 1, 1, 1, true)?;
        Ok(Self { conv1, conv2 })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = x.elu(1.0)?;
        let h = self.conv1.forward(&h)?;
        let h = h.elu(1.0)?;
        let h = self.conv2.forward(&h)?;
        Ok((h + residual)?)
    }
}

/// Mimi encoder transformer layer: LayerNorm(+bias) → causal MHA (sliding-window 250, RoPE θ1e4)
/// → LayerScale residual; LayerNorm → fc1→GELU→fc2 → LayerScale residual.
struct MimiEncoderLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    fc1: Linear,
    fc2: Linear,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    self_attn_layer_scale: LayerScale,
    mlp_layer_scale: LayerScale,
    num_heads: usize,
    head_dim: usize,
}

impl MimiEncoderLayer {
    fn new(vb: VarBuilder, cfg: &CodecEncoderConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            q_proj: linear_no_bias(
                h,
                cfg.num_attention_heads * cfg.head_dim,
                vb.pp("self_attn.q_proj"),
            )?,
            k_proj: linear_no_bias(
                h,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("self_attn.k_proj"),
            )?,
            v_proj: linear_no_bias(
                h,
                cfg.num_key_value_heads * cfg.head_dim,
                vb.pp("self_attn.v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                h,
                vb.pp("self_attn.o_proj"),
            )?,
            fc1: linear_no_bias(h, cfg.intermediate_size, vb.pp("mlp.fc1"))?,
            fc2: linear_no_bias(cfg.intermediate_size, h, vb.pp("mlp.fc2"))?,
            input_layernorm: layer_norm(h, cfg.norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: layer_norm(
                h,
                cfg.norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            self_attn_layer_scale: LayerScale::new(vb.pp("self_attn_layer_scale"), h)?,
            mlp_layer_scale: LayerScale::new(vb.pp("mlp_layer_scale"), h)?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
        })
    }
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let q = self
            .q_proj
            .forward(&h)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(&h)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&h)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin, true)?;
        let scaling = (self.head_dim as f64).powf(-0.5);
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn = (q.matmul(&k.t()?.contiguous()?)? * scaling)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.contiguous()?;
        let h = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t,
            self.num_heads * self.head_dim,
        ))?;
        let h = self.o_proj.forward(&h)?;
        let h = (residual + self.self_attn_layer_scale.forward(&h)?)?;
        let residual = h.clone();
        let m = self.post_attention_layernorm.forward(&h)?;
        let m = self.fc2.forward(&self.fc1.forward(&m)?.gelu()?)?;
        Ok((residual + self.mlp_layer_scale.forward(&m)?)?)
    }
}

/// One encode-side RVQ: input_proj (512→256), N codebooks; greedy residual nearest-neighbor.
struct RvqEncode {
    input_proj: Conv1d,     // 1×1, 512 → 256
    codebooks: Vec<Tensor>, // each (2048, 256) precomputed embed
}

impl RvqEncode {
    fn new(vb: VarBuilder, n_layers: usize) -> Result<Self> {
        let input_proj = conv1d_no_bias(512, 256, 1, Conv1dConfig::default(), vb.pp("input_proj"))?;
        let mut codebooks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let cb_vb = vb.pp("layers").pp(i).pp("codebook");
            let embed_sum = cb_vb.get((2048, 256), "embed_sum")?;
            let cluster_usage = cb_vb.get(2048, "cluster_usage")?;
            let embed = embed_sum.to_dtype(DType::F32)?.broadcast_div(
                &cluster_usage
                    .to_dtype(DType::F32)?
                    .clamp(1e-5, f64::INFINITY)?
                    .unsqueeze(1)?,
            )?;
            codebooks.push(embed);
        }
        Ok(Self {
            input_proj,
            codebooks,
        })
    }
    /// embeddings: (B, 512, T) → codes: Vec of (B, T), one per codebook.
    fn encode(&self, embeddings: &Tensor) -> Result<Vec<Tensor>> {
        let mut residual = self.input_proj.forward(embeddings)?; // (B, 256, T)
        let (b, _d, t) = residual.dims3()?;
        let mut out = Vec::with_capacity(self.codebooks.len());
        for embed in &self.codebooks {
            // nearest row by L2 via ||a||² + ||b||² - 2 a·b on (B*T, 256).
            let flat = residual.transpose(1, 2)?.reshape((b * t, 256))?; // (N, 256)
            let a2 = flat.sqr()?.sum_keepdim(1)?; // (N,1)
            let b2 = embed.sqr()?.sum_keepdim(1)?.t()?; // (1,2048)
            let dots = flat.matmul(&embed.t()?)?; // (N,2048)
            let dists = (a2.broadcast_add(&b2)? - dots * 2.0)?;
            let idx = dists.argmin(D::Minus1)?; // (N,)
            out.push(idx.reshape((b, t))?);
            let gathered = embed.index_select(&idx, 0)?; // (N,256)
            let gathered = gathered.reshape((b, t, 256))?.transpose(1, 2)?; // (B,256,T)
            residual = (residual - gathered)?;
        }
        Ok(out)
    }
}

pub struct CodecEncoder {
    cfg: CodecEncoderConfig,
    conv_in: CausalConv,
    stages: Vec<(MimiResnetBlock, CausalConv)>, // (resnet, downsample conv) per stride
    conv_out: CausalConv,
    layers: Vec<MimiEncoderLayer>,
    rotary: RoPE,
    downsample: Conv1d, // k4 s2, replicate pad
    semantic: RvqEncode,
    acoustic: RvqEncode,
}

impl CodecEncoder {
    pub fn new(
        vb: VarBuilder,
        cfg: &CodecEncoderConfig,
        valid_num_quantizers: usize,
    ) -> Result<Self> {
        let nf = cfg.num_filters; // 64
        // SEANet layout (confirmed weight names): conv_in at layers.0; per stage s in strides
        // [4,5,6,8]: resnet at layers.{1,4,7,10}.block, down conv at layers.{3,6,9,12}; conv_out at layers.14.
        let conv_in = CausalConv::new(
            vb.pp("encoder").pp("layers").pp(0),
            cfg.audio_channels,
            nf,
            cfg.kernel_size,
            1,
            1,
            1,
            true,
        )?;
        let strides: Vec<usize> = cfg.upsampling_ratios.iter().rev().cloned().collect(); // [4,5,6,8]
        let resnet_idx = [1usize, 4, 7, 10];
        let down_idx = [3usize, 6, 9, 12];
        let mut stages = Vec::with_capacity(strides.len());
        let mut dim = nf;
        for (s, (&ri, &di)) in strides.iter().zip(resnet_idx.iter().zip(down_idx.iter())) {
            let resnet =
                MimiResnetBlock::new(vb.pp("encoder").pp("layers").pp(ri).pp("block"), dim)?;
            let down = CausalConv::new(
                vb.pp("encoder").pp("layers").pp(di),
                dim,
                dim * 2,
                2 * s,
                *s,
                1,
                1,
                true,
            )?;
            stages.push((resnet, down));
            dim *= 2;
        }
        let conv_out = CausalConv::new(
            vb.pp("encoder").pp("layers").pp(14),
            dim,
            cfg.hidden_size,
            cfg.last_kernel_size,
            1,
            1,
            1,
            true,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(MimiEncoderLayer::new(
                vb.pp("encoder_transformer").pp("layers").pp(i),
                cfg,
            )?);
        }
        let rotary = RoPE::new(cfg.head_dim, cfg.rope_theta as f32, vb.device())?;

        // downsample: k4 s2, replicate pad (left = kernel-stride = 2). bias=False.
        let ds_w = vb
            .pp("downsample")
            .get((cfg.hidden_size, cfg.hidden_size, 4), "conv.weight")?;
        let downsample = Conv1d::new(
            ds_w,
            None,
            Conv1dConfig {
                padding: 0,
                stride: 2,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );

        let n_semantic = cfg.num_semantic_quantizers; // 1
        let n_acoustic = valid_num_quantizers - n_semantic; // 15
        let semantic = RvqEncode::new(
            vb.pp("quantizer").pp("semantic_residual_vector_quantizer"),
            n_semantic,
        )?;
        let acoustic = RvqEncode::new(
            vb.pp("quantizer").pp("acoustic_residual_vector_quantizer"),
            n_acoustic,
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            conv_in,
            stages,
            conv_out,
            layers,
            rotary,
            downsample,
            semantic,
            acoustic,
        })
    }

    /// wav: (B, 1, T_samples) f32 @ 24kHz → codes (B, 16, T_frames) u32 @ 12.5Hz.
    pub fn encode(&self, wav: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(wav)?;
        for (resnet, down) in &self.stages {
            h = resnet.forward(&h)?;
            h = h.elu(1.0)?;
            h = down.forward(&h)?;
        }
        h = h.elu(1.0)?;
        h = self.conv_out.forward(&h)?; // (B, 512, 25Hz)
        let mut z = h.transpose(1, 2)?; // (B, T, 512)
        let t = z.dim(1)?;
        let (cos, sin) = self.rotary.forward(0, t, z.device())?;
        let mask = sliding_window_causal_mask(t, self.cfg.sliding_window, z.device())?;
        for layer in &self.layers {
            z = layer.forward(&z, &cos, &sin, &mask)?;
        }
        let z = z.transpose(1, 2)?; // (B, 512, T)
        let z = self.downsample_replicate(&z)?; // (B, 512, 12.5Hz)
        let mut codes = self.semantic.encode(&z)?;
        codes.extend(self.acoustic.encode(&z)?);
        let stacked = Tensor::stack(&codes, 1)?; // (B, 16, T)
        Ok(stacked.to_dtype(DType::U32)?)
    }

    /// Mimi downsample conv: replicate left-pad (kernel-stride=2) + computed right extra-pad, k4 s2.
    fn downsample_replicate(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, t) = x.dims3()?;
        let left = 4 - 2; // kernel - stride
        let extra = conv1d_extra_padding(t, 4, left, 2);
        let first = x.narrow(D::Minus1, 0, 1)?.repeat((1, 1, left))?;
        let last = x.narrow(D::Minus1, t - 1, 1)?.repeat((1, 1, extra))?;
        let x = Tensor::cat(&[&first, x, &last], D::Minus1)?;
        Ok(self.downsample.forward(&x)?)
    }
}

// ---------------------------------------------------------------------------
// Combined tokenizer.
// ---------------------------------------------------------------------------

pub struct SpeechTokenizer {
    pub encoder: Option<CodecEncoder>,
    pub decoder: CodecDecoder,
    pub output_sample_rate: usize,
    pub frame_rate: f64,
}

impl SpeechTokenizer {
    /// Load both halves from the codec safetensors. `with_encoder` = voice cloning support.
    pub fn new(
        vb: VarBuilder,
        cfg: &crate::models::qwen3_tts::config::SpeechTokenizerConfig,
        with_encoder: bool,
    ) -> Result<Self> {
        let decoder = CodecDecoder::new(vb.pp("decoder"), &cfg.decoder_config)?;
        let encoder = if with_encoder {
            Some(CodecEncoder::new(
                vb.pp("encoder"),
                &cfg.encoder_config,
                cfg.encoder_valid_num_quantizers,
            )?)
        } else {
            None
        };
        Ok(Self {
            encoder,
            decoder,
            output_sample_rate: cfg.output_sample_rate,
            frame_rate: cfg.frame_rate(),
        })
    }
}
