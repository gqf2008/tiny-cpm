//! Mimi-style 12 Hz codec decoder on burn, ported 1:1 from tiny-cpm
//! src/models/qwen3_tts/codec.rs (decoder half only; the encoder is only
//! needed for voice cloning, out of scope for this spike). Runs in F32 like
//! the candle reference.
//!
//! EnCodec-exact causal-conv padding: left pad `kernel_eff - stride` zeros +
//! a computed right "extra padding" so the length stays divisible; transposed
//! convs trim `kernel - stride` samples from the right.

use anyhow::{Result, anyhow};
use burn::tensor::{
    DType, Int, Tensor, module,
    ops::{ConvOptions, ConvTransposeOptions},
};

use crate::config::{CodecDecoderConfig, CodecEncoderConfig};
use crate::model::{CODEC_DT, Weights, layer_norm, linear, repeat_kv, rms_norm};

fn embed(x: Tensor<3>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<3> {
    let w2 = w.swap_dims(0, 1).unsqueeze::<3>();
    let out = x.matmul(w2);
    match b {
        Some(b) => out + b.unsqueeze::<2>().unsqueeze::<3>(),
        None => out,
    }
}

// ---------------------------------------------------------------------------
// Causal conv helpers (EnCodec-exact padding)
// ---------------------------------------------------------------------------

/// `_get_extra_padding_for_conv1d`: right zero-pad so
/// `(len - k_eff + pad) / stride + 1` is integral, `pad = k_eff - stride`.
fn conv1d_extra_padding(len: usize, kernel_eff: usize, padding: usize, stride: usize) -> usize {
    let n_frames = (len as f64 - kernel_eff as f64 + padding as f64) / stride as f64 + 1.0;
    let ideal = (n_frames.ceil() as usize - 1) * stride + (kernel_eff - padding);
    ideal.saturating_sub(len)
}

/// Left zero-pad `left` + right zero-pad `right` on the last dim of (B, C, T).
fn pad_zeros_last_dim(
    x: Tensor<3>,
    left: usize,
    right: usize,
    device: &burn::tensor::Device,
) -> Tensor<3> {
    if left == 0 && right == 0 {
        return x;
    }
    let [b, c, _t] = x.dims();
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        parts.push(Tensor::<3, burn::tensor::Float>::zeros([b, c, left], device).cast(CODEC_DT));
    }
    parts.push(x);
    if right > 0 {
        parts.push(Tensor::<3, burn::tensor::Float>::zeros([b, c, right], device).cast(CODEC_DT));
    }
    Tensor::cat(parts, 2)
}

/// `Qwen3TTSTokenizerV2CausalConvNet`: left-pad + right extra-pad, then a plain
/// conv1d (no built-in padding). Weight name nested under `.conv`.
struct CausalConv {
    w: Tensor<3>,
    b: Option<Tensor<1>>,
    stride: usize,
    kernel_eff: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl CausalConv {
    fn new(
        w: &Weights,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let wt = w.get(&format!("{prefix}.conv.weight"), CODEC_DT)?;
        let b = if bias {
            Some(w.get(&format!("{prefix}.conv.bias"), CODEC_DT)?)
        } else {
            None
        };
        let kernel_eff = (k - 1) * dilation + 1;
        Ok(Self {
            w: wt,
            b,
            stride,
            kernel_eff,
            padding: kernel_eff - stride,
            dilation,
            groups,
        })
    }
    fn forward(&self, x: Tensor<3>, device: &burn::tensor::Device) -> Tensor<3> {
        let len = x.dims()[2];
        let extra = conv1d_extra_padding(len, self.kernel_eff, self.padding, self.stride);
        let x = pad_zeros_last_dim(x, self.padding, extra, device);
        let opts = ConvOptions::new([self.stride], [0], [self.dilation], self.groups);
        module::conv1d(x, self.w.clone(), self.b.clone(), opts)
    }
}

/// `Qwen3TTSTokenizerV2CausalTransConvNet`: plain conv_transpose1d then trim
/// `kernel - stride` samples from the right.
struct CausalTransConv {
    w: Tensor<3>, // (in, out, k)
    b: Option<Tensor<1>>,
    stride: usize,
    right_pad: usize,
}

impl CausalTransConv {
    fn new(
        w: &Weights,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Self> {
        let wt = w.get(&format!("{prefix}.conv.weight"), CODEC_DT)?;
        let b = if bias {
            Some(w.get(&format!("{prefix}.conv.bias"), CODEC_DT)?)
        } else {
            None
        };
        Ok(Self {
            w: wt,
            b,
            stride,
            right_pad: k.saturating_sub(stride),
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let opts = ConvTransposeOptions::new([self.stride], [0], [0], [1], 1);
        let y = module::conv_transpose1d(x, self.w.clone(), self.b.clone(), opts);
        if self.right_pad > 0 {
            let t = y.dims()[2];
            y.narrow(2, 0, t - self.right_pad)
        } else {
            y
        }
    }
}

// ---------------------------------------------------------------------------
// SnakeBeta activation
// ---------------------------------------------------------------------------

/// `SnakeBeta`: x + (1/(exp(beta)+1e-9)) * sin^2(x * exp(alpha)), per-channel.
struct SnakeBeta {
    alpha: Tensor<1>,
    beta: Tensor<1>,
}

impl SnakeBeta {
    fn new(w: &Weights, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            alpha: w.get(&format!("{prefix}.alpha"), CODEC_DT)?,
            beta: w.get(&format!("{prefix}.beta"), CODEC_DT)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, c, _] = x.dims();
        let alpha = self.alpha.clone().exp().reshape([1, c, 1]);
        let beta = self.beta.clone().exp().reshape([1, c, 1]);
        let periodic = (x.clone() * alpha).sin().powf_scalar(2.0) / (beta + 1e-9);
        x + periodic
    }
}

// ---------------------------------------------------------------------------
// Decoder RVQ (dequant)
// ---------------------------------------------------------------------------

/// One EMA Euclidean codebook, decode-side: embedding = embedding_sum /
/// clamp(cluster_usage, 1e-5).
struct EuclideanCodebookDecode {
    embedding: Tensor<2>, // (codebook_size, dim)
}

impl EuclideanCodebookDecode {
    fn new(w: &Weights, prefix: &str) -> Result<Self> {
        // NOTE: decoder files spell it `embedding_sum` (encoder uses `embed_sum`).
        let embedding_sum = w
            .get(&format!("{prefix}.embedding_sum"), CODEC_DT)
            .or_else(|_| w.get(&format!("{prefix}.embed_sum"), CODEC_DT))?;
        let cluster_usage: Tensor<1> = w.get(&format!("{prefix}.cluster_usage"), CODEC_DT)?;
        let usage = cluster_usage.clamp_min(1e-5).unsqueeze_dim::<2>(1);
        let embedding = embedding_sum / usage;
        Ok(Self { embedding })
    }
    /// codes (B, T) Int → (B, T, dim).
    fn decode(&self, codes: Tensor<2, Int>) -> Tensor<3> {
        let [b, t] = codes.dims();
        let flat = codes.reshape([b * t]);
        self.embedding
            .clone()
            .select(0, flat)
            .reshape([b, t, self.embedding.dims()[1]])
    }
}

/// `VectorQuantization.decode`: codebook decode → (B, dim, T).
struct VectorQuantizationDecode {
    codebook: EuclideanCodebookDecode,
}

impl VectorQuantizationDecode {
    fn decode(&self, codes: Tensor<2, Int>) -> Tensor<3> {
        let q = self.codebook.decode(codes); // (B, T, dim)
        q.swap_dims(1, 2) // (B, dim, T)
    }
}

/// `ResidualVectorQuantizer.decode`: sum per-layer dequant + 1×1 output_proj conv.
struct RvqDecode {
    layers: Vec<VectorQuantizationDecode>,
    output_proj_w: Tensor<3>, // (512, 256, 1)
}

impl RvqDecode {
    fn new(w: &Weights, prefix: &str, n_layers: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(VectorQuantizationDecode {
                codebook: EuclideanCodebookDecode::new(
                    w,
                    &format!("{prefix}.vq.layers.{i}._codebook"),
                )?,
            });
        }
        let output_proj_w = w.get(&format!("{prefix}.output_proj.weight"), CODEC_DT)?;
        Ok(Self {
            layers,
            output_proj_w,
        })
    }
    /// codes: slice of per-layer (B, T) → (B, 512, T)
    fn decode(&self, codes: &[Tensor<2, Int>]) -> Result<Tensor<3>> {
        if codes.len() != self.layers.len() {
            return Err(anyhow!("rvq layer count mismatch"));
        }
        let mut acc: Option<Tensor<3>> = None;
        for (layer, codes_i) in self.layers.iter().zip(codes.iter()) {
            let q = layer.decode(codes_i.clone()); // (B, dim, T)
            acc = Some(match acc {
                None => q,
                Some(a) => a + q,
            });
        }
        let q = acc.ok_or_else(|| anyhow!("rvq: no layers"))?;
        // 1×1 conv: (B, 256, T) → (B, 512, T)
        let opts = ConvOptions::new([1], [0], [1], 1);
        Ok(module::conv1d(q, self.output_proj_w.clone(), None, opts))
    }
}

/// `SplitResidualVectorQuantizer.decode` (decoder side): rvq_first (1 semantic)
/// + rvq_rest (15 acoustic).
struct SplitRvqDecode {
    rvq_first: RvqDecode,
    rvq_rest: RvqDecode,
}

impl SplitRvqDecode {
    fn new(w: &Weights) -> Result<Self> {
        Ok(Self {
            rvq_first: RvqDecode::new(w, "decoder.quantizer.rvq_first", 1)?,
            rvq_rest: RvqDecode::new(w, "decoder.quantizer.rvq_rest", 15)?,
        })
    }
    /// codes (B, 16, T) Int → (B, 512, T)
    fn decode(&self, codes: &Tensor<3, Int>) -> Result<Tensor<3>> {
        let b = codes.dims()[0];
        let t = codes.dims()[2];
        let first = vec![codes.clone().narrow(1, 0, 1).reshape([b, t])];
        let mut rest = Vec::with_capacity(15);
        for i in 1..16 {
            rest.push(codes.clone().narrow(1, i, 1).reshape([b, t]));
        }
        let q = self.rvq_first.decode(&first)?;
        let q2 = self.rvq_rest.decode(&rest)?;
        Ok(q + q2)
    }
}

// ---------------------------------------------------------------------------
// Decoder pre-transformer (RMSNorm, no qk-norm, sliding-window 72, gated SiLU,
// LayerScale, RoPE θ1e4). Full attention with mask, no KV cache.
// ---------------------------------------------------------------------------

struct LayerScale {
    scale: Tensor<1>,
}

impl LayerScale {
    fn new(w: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            scale: w.get(&format!("{prefix}.scale"), CODEC_DT)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        x * self.scale.clone().unsqueeze::<2>().unsqueeze::<3>()
    }
}

/// Sliding-window causal mask: query i attends keys [i-window+1, i].
/// Additive (1,1,T,T) CODEC_DT.
fn sliding_window_causal_mask(t: usize, window: usize, device: &burn::tensor::Device) -> Tensor<4> {
    let mut m = vec![f32::NEG_INFINITY; t * t];
    for i in 0..t {
        let lo = i.saturating_sub(window - 1);
        for j in lo..=i {
            m[i * t + j] = 0.0;
        }
    }
    Tensor::<1>::from_floats(m.as_slice(), device)
        .reshape([1, 1, t, t])
        .cast(CODEC_DT)
}

struct DecoderAttention {
    q_w: Tensor<2>,
    k_w: Tensor<2>,
    v_w: Tensor<2>,
    o_w: Tensor<2>,
    num_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
    scaling: f64,
}

impl DecoderAttention {
    fn new(w: &Weights, prefix: &str, cfg: &CodecDecoderConfig) -> Result<Self> {
        let nh = cfg.num_attention_heads;
        let hd = cfg.head_dim;
        let kvh = cfg.num_key_value_heads;
        Ok(Self {
            q_w: w.get(&format!("{prefix}.self_attn.q_proj.weight"), CODEC_DT)?,
            k_w: w.get(&format!("{prefix}.self_attn.k_proj.weight"), CODEC_DT)?,
            v_w: w.get(&format!("{prefix}.self_attn.v_proj.weight"), CODEC_DT)?,
            o_w: w.get(&format!("{prefix}.self_attn.o_proj.weight"), CODEC_DT)?,
            num_heads: nh,
            head_dim: hd,
            num_kv_groups: nh / kvh,
            scaling: (hd as f64).powf(-0.5),
        })
    }

    /// x: (B, T, hidden). Full attention with sliding-window causal mask.
    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        mask: &Tensor<4>,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let kv_heads = self.num_heads / self.num_kv_groups;
        let q = embed(x.clone(), self.q_w.clone(), None)
            .reshape([b, t, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = embed(x.clone(), self.k_w.clone(), None)
            .reshape([b, t, kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = embed(x, self.v_w.clone(), None)
            .reshape([b, t, kv_heads, self.head_dim])
            .swap_dims(1, 2);
        // q/k norm are Identity in this decoder; apply RoPE (θ1e4). Everything
        // is F32 already, so no tof32 casts needed.
        let (q, k) = apply_rope_codec(q, k, cos, sin);
        let k = repeat_kv(k, self.num_kv_groups);
        let v = repeat_kv(v, self.num_kv_groups);
        let attn = q.matmul(k.swap_dims(2, 3)) * self.scaling;
        let attn = attn + mask.clone();
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v); // (B, H, T, hd)
        let out = out
            .swap_dims(1, 2)
            .reshape([b, t, self.num_heads * self.head_dim]);
        embed(out, self.o_w.clone(), None)
    }
}

/// GPT-NeoX split-half rotation for the codec (rank-2 cos/sin, F32).
fn apply_rope_codec(
    q: Tensor<4>,
    k: Tensor<4>,
    cos: &Tensor<2>,
    sin: &Tensor<2>,
) -> (Tensor<4>, Tensor<4>) {
    let cos = cos.clone().unsqueeze::<4>();
    let sin = sin.clone().unsqueeze::<4>();
    let half = q.dims()[3] / 2;
    let rot = |x: Tensor<4>| -> Tensor<4> {
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.clone().narrow(3, half, half);
        Tensor::cat(vec![x2.neg(), x1], 3)
    };
    (
        q.clone() * cos.clone() + rot(q.clone()) * sin.clone(),
        k.clone() * cos + rot(k) * sin,
    )
}

struct DecoderTransformerLayer {
    attn: DecoderAttention,
    gate_w: Tensor<2>,
    up_w: Tensor<2>,
    down_w: Tensor<2>,
    in_ln_w: Tensor<1>,
    post_ln_w: Tensor<1>,
    attn_scale: LayerScale,
    mlp_scale: LayerScale,
    eps: f64,
}

impl DecoderTransformerLayer {
    fn new(w: &Weights, prefix: &str, cfg: &CodecDecoderConfig) -> Result<Self> {
        Ok(Self {
            attn: DecoderAttention::new(w, prefix, cfg)?,
            gate_w: w.get(&format!("{prefix}.mlp.gate_proj.weight"), CODEC_DT)?,
            up_w: w.get(&format!("{prefix}.mlp.up_proj.weight"), CODEC_DT)?,
            down_w: w.get(&format!("{prefix}.mlp.down_proj.weight"), CODEC_DT)?,
            in_ln_w: w.get(&format!("{prefix}.input_layernorm.weight"), CODEC_DT)?,
            post_ln_w: w.get(
                &format!("{prefix}.post_attention_layernorm.weight"),
                CODEC_DT,
            )?,
            attn_scale: LayerScale::new(
                w,
                &format!("{prefix}.self_attn_layer_scale"),
                cfg.hidden_size,
            )?,
            mlp_scale: LayerScale::new(w, &format!("{prefix}.mlp_layer_scale"), cfg.hidden_size)?,
            eps: cfg.rms_norm_eps,
        })
    }
    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        mask: &Tensor<4>,
    ) -> Tensor<3> {
        let residual = x.clone();
        let h = rms_norm(x, self.in_ln_w.clone(), self.eps);
        let h = self.attn.forward(h, cos, sin, mask);
        let h = residual + self.attn_scale.forward(h);
        let residual = h.clone();
        let m = rms_norm(h, self.post_ln_w.clone(), self.eps);
        let gate = burn::tensor::activation::silu(embed(m.clone(), self.gate_w.clone(), None));
        let up = embed(m, self.up_w.clone(), None);
        let m = embed(gate * up, self.down_w.clone(), None);
        residual + self.mlp_scale.forward(m)
    }
}

struct DecoderPreTransformer {
    input_proj_w: Tensor<2>,
    input_proj_b: Tensor<1>,
    output_proj_w: Tensor<2>,
    output_proj_b: Tensor<1>,
    layers: Vec<DecoderTransformerLayer>,
    norm_w: Tensor<1>,
    inv_freq: Vec<f32>,
    sliding_window: usize,
    eps: f64,
    device: burn::tensor::Device,
}

impl DecoderPreTransformer {
    fn new(w: &Weights, root: &str, cfg: &CodecDecoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderTransformerLayer::new(
                w,
                &format!("{root}.pre_transformer.layers.{i}"),
                cfg,
            )?);
        }
        Ok(Self {
            input_proj_w: w.get(
                &format!("{root}.pre_transformer.input_proj.weight"),
                CODEC_DT,
            )?,
            input_proj_b: w.get(&format!("{root}.pre_transformer.input_proj.bias"), CODEC_DT)?,
            output_proj_w: w.get(
                &format!("{root}.pre_transformer.output_proj.weight"),
                CODEC_DT,
            )?,
            output_proj_b: w.get(
                &format!("{root}.pre_transformer.output_proj.bias"),
                CODEC_DT,
            )?,
            layers,
            norm_w: w.get(&format!("{root}.pre_transformer.norm.weight"), CODEC_DT)?,
            inv_freq: crate::model::compute_default_rope_parameters(
                cfg.head_dim,
                cfg.rope_theta as f32,
            ),
            sliding_window: cfg.sliding_window,
            eps: cfg.rms_norm_eps,
            device: w.device(),
        })
    }
    /// x: (B, T, latent) → (B, T, latent)
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mut h = embed(
            x,
            self.input_proj_w.clone(),
            Some(self.input_proj_b.clone()),
        );
        let t = h.dims()[1];
        let (cos, sin) = crate::model::rotary(&self.device, &self.inv_freq, 0, t, CODEC_DT);
        let mask = sliding_window_causal_mask(t, self.sliding_window, &self.device);
        for layer in &self.layers {
            h = layer.forward(h, &cos, &sin, &mask);
        }
        let h = rms_norm(h, self.norm_w.clone(), self.eps);
        embed(
            h,
            self.output_proj_w.clone(),
            Some(self.output_proj_b.clone()),
        )
    }
}

// ---------------------------------------------------------------------------
// ConvNeXt block (upsample) + waveform decoder blocks
// ---------------------------------------------------------------------------

struct ConvNeXtBlock {
    dwconv: CausalConv,
    norm_w: Tensor<1>,
    norm_b: Tensor<1>,
    pwconv1_w: Tensor<2>,
    pwconv1_b: Tensor<1>,
    pwconv2_w: Tensor<2>,
    pwconv2_b: Tensor<1>,
    gamma: Tensor<1>,
    dim: usize,
}

impl ConvNeXtBlock {
    fn new(w: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            dwconv: CausalConv::new(w, &format!("{prefix}.dwconv"), dim, dim, 7, 1, 1, dim, true)?,
            norm_w: w.get(&format!("{prefix}.norm.weight"), CODEC_DT)?,
            norm_b: w.get(&format!("{prefix}.norm.bias"), CODEC_DT)?,
            pwconv1_w: w.get(&format!("{prefix}.pwconv1.weight"), CODEC_DT)?,
            pwconv1_b: w.get(&format!("{prefix}.pwconv1.bias"), CODEC_DT)?,
            pwconv2_w: w.get(&format!("{prefix}.pwconv2.weight"), CODEC_DT)?,
            pwconv2_b: w.get(&format!("{prefix}.pwconv2.bias"), CODEC_DT)?,
            gamma: w.get(&format!("{prefix}.gamma"), CODEC_DT)?,
            dim,
        })
    }
    /// x: (B, C, T)
    fn forward(&self, x: Tensor<3>, device: &burn::tensor::Device) -> Tensor<3> {
        let input = x.clone();
        let h = self.dwconv.forward(x, device);
        let h = h.swap_dims(1, 2); // (B, T, C)
        let h = layer_norm(h, self.norm_w.clone(), self.norm_b.clone(), 1e-6);
        let h = embed(h, self.pwconv1_w.clone(), Some(self.pwconv1_b.clone()));
        let h = burn::tensor::activation::gelu(h);
        let h = embed(h, self.pwconv2_w.clone(), Some(self.pwconv2_b.clone()));
        let h = h * self.gamma.clone().unsqueeze::<2>().unsqueeze::<3>();
        let h = h.swap_dims(1, 2); // (B, C, T)
        input + h
    }
}

struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv,
    act2: SnakeBeta,
    conv2: CausalConv,
}

impl DecoderResidualUnit {
    fn new(w: &Weights, prefix: &str, dim: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(w, &format!("{prefix}.act1"), dim)?,
            conv1: CausalConv::new(
                w,
                &format!("{prefix}.conv1"),
                dim,
                dim,
                7,
                1,
                dilation,
                1,
                true,
            )?,
            act2: SnakeBeta::new(w, &format!("{prefix}.act2"), dim)?,
            conv2: CausalConv::new(w, &format!("{prefix}.conv2"), dim, dim, 1, 1, 1, 1, true)?,
        })
    }
    fn forward(&self, x: Tensor<3>, device: &burn::tensor::Device) -> Tensor<3> {
        let residual = x.clone();
        let h = self.act1.forward(x);
        let h = self.conv1.forward(h, device);
        let h = self.act2.forward(h);
        let h = self.conv2.forward(h, device);
        h + residual
    }
}

/// One `Qwen3TTSTokenizerV2DecoderDecoderBlock`: SnakeBeta → ConvTranspose →
/// 3 residual units (dil 1, 3, 9).
struct DecoderBlock {
    snake: SnakeBeta,
    trans_conv: CausalTransConv,
    residuals: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    fn new(w: &Weights, root: &str, cfg: &CodecDecoderConfig, layer_idx: usize) -> Result<Self> {
        let in_dim = cfg.decoder_dim / 2usize.pow(layer_idx as u32);
        let out_dim = cfg.decoder_dim / 2usize.pow(layer_idx as u32 + 1);
        let rate = cfg.upsample_rates[layer_idx];
        let snake = SnakeBeta::new(
            w,
            &format!("{root}.decoder.{}.block.0", layer_idx + 1),
            in_dim,
        )?;
        let trans_conv = CausalTransConv::new(
            w,
            &format!("{root}.decoder.{}.block.1", layer_idx + 1),
            in_dim,
            out_dim,
            2 * rate,
            rate,
            true,
        )?;
        let mut residuals = Vec::with_capacity(3);
        for (ri, dilation) in [1usize, 3, 9].iter().enumerate() {
            residuals.push(DecoderResidualUnit::new(
                w,
                &format!("{root}.decoder.{}.block.{}", layer_idx + 1, ri + 2),
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
    fn forward(&self, x: Tensor<3>, device: &burn::tensor::Device) -> Tensor<3> {
        let mut h = self.snake.forward(x);
        h = self.trans_conv.forward(h);
        for r in &self.residuals {
            h = r.forward(h, device);
        }
        h
    }
}

// ---------------------------------------------------------------------------
// CodecDecoder (public)
// ---------------------------------------------------------------------------

pub struct CodecDecoder {
    quantizer: SplitRvqDecode,
    pre_conv: CausalConv,
    pre_transformer: DecoderPreTransformer,
    upsample: Vec<(CausalTransConv, ConvNeXtBlock)>,
    decoder0: CausalConv,
    blocks: Vec<DecoderBlock>,
    final_snake: SnakeBeta,
    final_conv: CausalConv,
    total_upsample: usize,
    device: burn::tensor::Device,
}

impl CodecDecoder {
    pub fn new(w: &Weights, cfg: &CodecDecoderConfig) -> Result<Self> {
        let quantizer = SplitRvqDecode::new(w)?;
        let pre_conv = CausalConv::new(
            w,
            "decoder.pre_conv",
            cfg.codebook_dim,
            cfg.latent_dim,
            3,
            1,
            1,
            1,
            true,
        )?;
        let pre_transformer = DecoderPreTransformer::new(w, "decoder", cfg)?;
        let mut upsample = Vec::with_capacity(cfg.upsampling_ratios.len());
        for (i, &factor) in cfg.upsampling_ratios.iter().enumerate() {
            let tc = CausalTransConv::new(
                w,
                &format!("decoder.upsample.{i}.0"),
                cfg.latent_dim,
                cfg.latent_dim,
                factor,
                factor,
                true,
            )?;
            let cn = ConvNeXtBlock::new(w, &format!("decoder.upsample.{i}.1"), cfg.latent_dim)?;
            upsample.push((tc, cn));
        }
        let decoder0 = CausalConv::new(
            w,
            "decoder.decoder.0",
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
            blocks.push(DecoderBlock::new(w, "decoder", cfg, i)?);
        }
        let n = cfg.upsample_rates.len();
        let out_dim = cfg.decoder_dim / 2usize.pow(n as u32);
        let final_snake = SnakeBeta::new(w, &format!("decoder.decoder.{}", n + 1), out_dim)?;
        let final_conv = CausalConv::new(
            w,
            &format!("decoder.decoder.{}", n + 2),
            out_dim,
            1,
            7,
            1,
            1,
            1,
            true,
        )?;
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
            device: w.device(),
        })
    }

    /// codes: (B, 16, T) Int → waveform (B, 1, T*1920) CODEC_DT in [-1, 1].
    pub fn decode(&self, codes: &Tensor<3, Int>) -> Result<Tensor<3>> {
        let hidden = self.quantizer.decode(codes)?; // (B, 512, T)
        let hidden = self.pre_conv.forward(hidden, &self.device).swap_dims(1, 2); // (B, T, latent)
        let hidden = self.pre_transformer.forward(hidden); // (B, T, latent)
        let mut hidden = hidden.swap_dims(1, 2); // (B, latent, T)
        for (tc, cn) in &self.upsample {
            hidden = tc.forward(hidden);
            hidden = cn.forward(hidden, &self.device);
        }
        let mut wav = self.decoder0.forward(hidden, &self.device);
        for block in &self.blocks {
            wav = block.forward(wav, &self.device);
        }
        wav = self.final_snake.forward(wav);
        wav = self.final_conv.forward(wav, &self.device);
        Ok(wav.clamp(-1.0, 1.0))
    }

    pub fn frame_samples(&self) -> usize {
        self.total_upsample
    }

    /// `chunked_decode`: `chunk_size`-frame chunks with `left_context`-frame
    /// left context (context output trimmed). codes (B, 16, T) → (B, 1, T*1920).
    pub fn chunked_decode(
        &self,
        codes: &Tensor<3, Int>,
        chunk_size: usize,
        left_context: usize,
    ) -> Result<Tensor<3>> {
        let t_total = codes.dims()[2];
        let mut wavs = Vec::new();
        let mut start = 0usize;
        while start < t_total {
            let end = (start + chunk_size).min(t_total);
            let context = if start > left_context {
                left_context
            } else {
                start
            };
            let chunk = codes
                .clone()
                .narrow(2, start - context, (end - start) + context);
            let wav = self.decode(&chunk)?;
            let drop = context * self.total_upsample;
            let t = wav.dims()[2];
            let wav = wav.narrow(2, drop, t - drop);
            wavs.push(wav);
            start = end;
        }
        Ok(Tensor::cat(wavs, 2))
    }
}

// ---------------------------------------------------------------------------
// CodecEncoder (voice cloning): Mimi SEANet + transformer + split-RVQ encode.
// Ported 1:1 from tiny-cpm src/models/qwen3_tts/codec.rs (encoder half), F32.
// ---------------------------------------------------------------------------

/// Mimi residual block: ELU → conv k3 (dim→dim/2) → ELU → conv k1 (dim/2→dim),
/// residual add. Convs at `...block.1` and `...block.3` (block.0/2 are ELU).
struct MimiResnetBlock {
    conv1: CausalConv,
    conv2: CausalConv,
}

impl MimiResnetBlock {
    fn new(w: &Weights, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv::new(w, &format!("{prefix}.1"), dim, dim / 2, 3, 1, 1, 1, true)?,
            conv2: CausalConv::new(w, &format!("{prefix}.3"), dim / 2, dim, 1, 1, 1, 1, true)?,
        })
    }
    fn forward(&self, x: Tensor<3>, device: &burn::tensor::Device) -> Tensor<3> {
        let residual = x.clone();
        let h = burn::tensor::activation::elu(x, 1.0);
        let h = self.conv1.forward(h, device);
        let h = burn::tensor::activation::elu(h, 1.0);
        let h = self.conv2.forward(h, device);
        h + residual
    }
}

/// Mimi encoder transformer layer: LayerNorm(+bias) → causal MHA (sliding-window
/// 250, RoPE θ1e4) → LayerScale residual; LayerNorm → fc1→GELU→fc2 → LayerScale.
struct MimiEncoderLayer {
    q_w: Tensor<2>,
    k_w: Tensor<2>,
    v_w: Tensor<2>,
    o_w: Tensor<2>,
    fc1_w: Tensor<2>,
    fc2_w: Tensor<2>,
    in_ln_w: Tensor<1>,
    in_ln_b: Tensor<1>,
    post_ln_w: Tensor<1>,
    post_ln_b: Tensor<1>,
    attn_scale: LayerScale,
    mlp_scale: LayerScale,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    eps: f64,
}

impl MimiEncoderLayer {
    fn new(w: &Weights, prefix: &str, cfg: &CodecEncoderConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            q_w: w.get(&format!("{prefix}.self_attn.q_proj.weight"), CODEC_DT)?,
            k_w: w.get(&format!("{prefix}.self_attn.k_proj.weight"), CODEC_DT)?,
            v_w: w.get(&format!("{prefix}.self_attn.v_proj.weight"), CODEC_DT)?,
            o_w: w.get(&format!("{prefix}.self_attn.o_proj.weight"), CODEC_DT)?,
            fc1_w: w.get(&format!("{prefix}.mlp.fc1.weight"), CODEC_DT)?,
            fc2_w: w.get(&format!("{prefix}.mlp.fc2.weight"), CODEC_DT)?,
            in_ln_w: w.get(&format!("{prefix}.input_layernorm.weight"), CODEC_DT)?,
            in_ln_b: w.get(&format!("{prefix}.input_layernorm.bias"), CODEC_DT)?,
            post_ln_w: w.get(
                &format!("{prefix}.post_attention_layernorm.weight"),
                CODEC_DT,
            )?,
            post_ln_b: w.get(&format!("{prefix}.post_attention_layernorm.bias"), CODEC_DT)?,
            attn_scale: LayerScale::new(
                w,
                &format!("{prefix}.self_attn_layer_scale"),
                cfg.hidden_size,
            )?,
            mlp_scale: LayerScale::new(w, &format!("{prefix}.mlp_layer_scale"), cfg.hidden_size)?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            scaling: (cfg.head_dim as f64).powf(-0.5),
            eps: cfg.norm_eps,
        })
    }
    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<2>,
        sin: &Tensor<2>,
        mask: &Tensor<4>,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let residual = x.clone();
        // LayerNorm(+bias); num_kv_heads == num_heads here (8/8), so k/v use num_heads.
        let h = layer_norm(x, self.in_ln_w.clone(), self.in_ln_b.clone(), self.eps);
        let q = embed(h.clone(), self.q_w.clone(), None)
            .reshape([b, t, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = embed(h.clone(), self.k_w.clone(), None)
            .reshape([b, t, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = embed(h, self.v_w.clone(), None)
            .reshape([b, t, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let (q, k) = apply_rope_codec(q, k, cos, sin);
        let attn = q.matmul(k.swap_dims(2, 3)) * self.scaling;
        let attn = attn + mask.clone();
        let attn = burn::tensor::activation::softmax(attn, 3);
        let h = attn
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, t, self.num_heads * self.head_dim]);
        let h = embed(h, self.o_w.clone(), None);
        let h = residual + self.attn_scale.forward(h);
        let residual = h.clone();
        let m = layer_norm(h, self.post_ln_w.clone(), self.post_ln_b.clone(), self.eps);
        let m = burn::tensor::activation::gelu(embed(m, self.fc1_w.clone(), None));
        let m = embed(m, self.fc2_w.clone(), None);
        residual + self.mlp_scale.forward(m)
    }
}

/// One encode-side RVQ: input_proj (512→256), N codebooks; greedy residual
/// nearest-neighbor by L2 (`||a||² + ||b||² - 2a·b`).
struct RvqEncode {
    input_proj_w: Tensor<3>, // (256, 512, 1)
    codebooks: Vec<Tensor<2>>,
}

impl RvqEncode {
    fn new(w: &Weights, prefix: &str, n_layers: usize) -> Result<Self> {
        let input_proj_w = w.get(&format!("{prefix}.input_proj.weight"), CODEC_DT)?;
        let mut codebooks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let cb = format!("{prefix}.layers.{i}.codebook");
            // NOTE: encoder files spell it `embed_sum` (decoder uses `embedding_sum`).
            let embed_sum: Tensor<2> = w.get(&format!("{cb}.embed_sum"), CODEC_DT)?;
            let cluster_usage: Tensor<1> = w.get(&format!("{cb}.cluster_usage"), CODEC_DT)?;
            let usage = cluster_usage.clamp_min(1e-5).unsqueeze_dim::<2>(1);
            codebooks.push(embed_sum / usage);
        }
        Ok(Self {
            input_proj_w,
            codebooks,
        })
    }
    /// embeddings (B, 512, T) → codes (B, T), one per codebook.
    fn encode(&self, embeddings: Tensor<3>, device: &burn::tensor::Device) -> Vec<Tensor<2, Int>> {
        let opts = ConvOptions::new([1], [0], [1], 1);
        let mut residual = module::conv1d(embeddings, self.input_proj_w.clone(), None, opts); // (B, 256, T)
        let [b, _d, t] = residual.dims();
        let mut out = Vec::with_capacity(self.codebooks.len());
        for embed in &self.codebooks {
            let flat = residual.clone().swap_dims(1, 2).reshape([b * t, 256]); // (N, 256)
            let a2 = flat.clone().powf_scalar(2.0).sum_dim(1); // (N, 1)
            let b2 = embed.clone().powf_scalar(2.0).sum_dim(1).swap_dims(0, 1); // (1, 2048)
            let dots = flat.clone().matmul(embed.clone().swap_dims(0, 1)); // (N, 2048)
            let dists = a2 + b2 - dots * 2.0; // broadcast (N,1)+(1,2048)
            let idx = dists.argmin(1).reshape([b * t]); // (N,)
            let gathered = embed.clone().select(0, idx.clone()).reshape([b, t, 256]); // (N,256)→(B,T,256)
            out.push(idx.reshape([b, t]));
            residual = residual - gathered.swap_dims(1, 2); // (B, 256, T)
        }
        out
    }
}

/// SEANet encoder: conv k7 → 4× (resnet + downsample conv k=2s s=s) → conv k3 →
/// 8-layer transformer → replicate-pad downsample k4 s2 → split-RVQ (1 semantic
/// + 15 acoustic).
pub struct CodecEncoder {
    conv_in: CausalConv,
    stages: Vec<(MimiResnetBlock, CausalConv)>,
    conv_out: CausalConv,
    layers: Vec<MimiEncoderLayer>,
    inv_freq: Vec<f32>,
    downsample_w: Tensor<3>, // (512, 512, 4)
    semantic: RvqEncode,
    acoustic: RvqEncode,
    sliding_window: usize,
    device: burn::tensor::Device,
}

impl CodecEncoder {
    pub fn new(w: &Weights, cfg: &CodecEncoderConfig, valid_num_quantizers: usize) -> Result<Self> {
        let device = w.device();
        let nf = cfg.num_filters; // 64
        // SEANet layout (weight names): conv_in at layers.0; per stride s in
        // [4,5,6,8]: resnet at layers.{1,4,7,10}.block, down at layers.{3,6,9,12};
        // conv_out at layers.14.
        let conv_in = CausalConv::new(
            w,
            "encoder.encoder.layers.0",
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
                MimiResnetBlock::new(w, &format!("encoder.encoder.layers.{ri}.block"), dim)?;
            let down = CausalConv::new(
                w,
                &format!("encoder.encoder.layers.{di}"),
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
            w,
            "encoder.encoder.layers.14",
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
                w,
                &format!("encoder.encoder_transformer.layers.{i}"),
                cfg,
            )?);
        }
        let inv_freq =
            crate::model::compute_default_rope_parameters(cfg.head_dim, cfg.rope_theta as f32);
        let downsample_w = w.get("encoder.downsample.conv.weight", CODEC_DT)?; // (512, 512, 4)
        let n_semantic = cfg.num_semantic_quantizers; // 1
        let n_acoustic = valid_num_quantizers - n_semantic; // 15
        let semantic = RvqEncode::new(
            w,
            "encoder.quantizer.semantic_residual_vector_quantizer",
            n_semantic,
        )?;
        let acoustic = RvqEncode::new(
            w,
            "encoder.quantizer.acoustic_residual_vector_quantizer",
            n_acoustic,
        )?;
        Ok(Self {
            conv_in,
            stages,
            conv_out,
            layers,
            inv_freq,
            downsample_w,
            semantic,
            acoustic,
            sliding_window: cfg.sliding_window,
            device,
        })
    }

    /// wav (B, 1, T_samples) f32 @ 24 kHz → codes (B, 16, T) Int @ 12.5 Hz.
    pub fn encode(&self, wav: Tensor<3>) -> Result<Tensor<3, Int>> {
        let mut h = self.conv_in.forward(wav, &self.device);
        for (resnet, down) in &self.stages {
            h = resnet.forward(h, &self.device);
            h = burn::tensor::activation::elu(h, 1.0);
            h = down.forward(h, &self.device);
        }
        h = burn::tensor::activation::elu(h, 1.0);
        h = self.conv_out.forward(h, &self.device); // (B, 512, 25 Hz)
        let mut z = h.swap_dims(1, 2); // (B, T, 512)
        let t = z.dims()[1];
        let mask = sliding_window_causal_mask(t, self.sliding_window, &self.device);
        let (cos, sin) = crate::model::rotary(&self.device, &self.inv_freq, 0, t, CODEC_DT);
        for layer in &self.layers {
            z = layer.forward(z, &cos, &sin, &mask);
        }
        let z = z.swap_dims(1, 2); // (B, 512, T)
        let z = self.downsample_replicate(z); // (B, 512, 12.5 Hz)
        let mut codes = self.semantic.encode(z.clone(), &self.device);
        codes.extend(self.acoustic.encode(z, &self.device));
        // stack (16 per-layer (B,T)) → (B, 16, T)
        let stacked: Tensor<3, Int> = Tensor::stack(codes, 1);
        Ok(stacked)
    }

    /// Mimi downsample conv: replicate left-pad (kernel-stride=2) + computed
    /// right extra-pad, k4 s2, no bias.
    fn downsample_replicate(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, _, t] = x.dims();
        let left = 4 - 2; // kernel - stride
        let extra = conv1d_extra_padding(t, 4, left, 2);
        let first = x.clone().narrow(2, 0, 1).repeat_dim(2, left);
        let last = x.clone().narrow(2, t - 1, 1).repeat_dim(2, extra);
        let x = Tensor::cat(vec![first, x, last], 2);
        let opts = ConvOptions::new([2], [0], [1], 1);
        module::conv1d(x, self.downsample_w.clone(), None, opts)
    }
}
