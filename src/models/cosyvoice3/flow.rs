//! CosyVoice3 Flow (DiT-CFM mel generator), ported from CrispASR's C++/ggml
//! implementation (github.com/CrispStrobe/CrispASR, src/cosyvoice3_tts.cpp;
//! upstream: cosyvoice/flow/{flow.py,flow_matching.py,DiT/dit.py,DiT/modules.py}).
//!
//! Pipeline per synthesis: speech tokens -> input_embedding lookup ->
//! PreLookahead conv stack -> repeat_interleave(token_mel_ratio) => mu;
//! speaker embedding -> L2-normalize -> spk_affine => spks; ref mel fills the
//! leading frames of cond (zeros after). The DiT estimator (22 AdaLN-Zero
//! blocks, bidirectional MHA with partial interleaved RoPE) predicts the CFM
//! velocity; a cosine-schedule Euler ODE with classifier-free guidance walks
//! seeded gaussian noise to the target mel.
//!
//! Weights load from the raw upstream `flow.pt` pickle (names kept verbatim,
//! e.g. `decoder.estimator.transformer_blocks.0.attn.to_q.weight`).
//!
//! When `cosyvoice3-flow-q8_0.gguf` exists in the model dir it takes
//! precedence: every tensor is dequantized to F16 and remapped to the flow.pt
//! name tree (GGUF keys `cosyvoice3.flow.*`, see CrispASR
//! convert-cosyvoice3-to-gguf.py), so the whole estimator runs F16 on Metal.
//! Hardcoded F32 tensors (RoPE tables, sinusoidal time embedding, ODE noise,
//! LayerNorm constants) are cast to the module dtype via `self.dtype`.
//!
//! CFG note: like CrispASR's non-batched fallback we run the estimator TWICE
//! per Euler step (cond + uncond, both T-major) instead of one B=2 forward —
//! numerically identical, simpler shapes.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::quantized::gguf_file;
use candle_core::{D, DType, Device, Tensor, pickle::read_all_with_key};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder, conv1d, linear, ops};
use rand::{RngExt, SeedableRng};

use crate::common::modules::eager_attention_forward;
use crate::position_embed::rope::{RoPE, rotate_half_interleave};

// Hparams (cosyvoice3.yaml flow section / CrispASR cv3_flow_hp).
const N_DIT_LAYERS: usize = 22;
const DIT_DIM: usize = 1024;
const DIT_HEADS: usize = 16;
const DIT_HEAD_DIM: usize = 64;
const DIT_FF_DIM: usize = 2048;
const DIT_INPUT_DIM: usize = 320; // 4 * mel: cat[x, cond, mu, spks]
const MEL_DIM: usize = 80;
const SPK_DIM_IN: usize = 192;
const SPEECH_CODEBOOK: usize = 6561;
const PRE_LOOKAHEAD_LEN: usize = 3;
/// DiT chunk-causal mask granularity in mel frames (upstream
/// cosyvoice3.yaml `static_chunk_size`; DiT/dit.py:163-166).
const STATIC_CHUNK_SIZE: usize = 50;
const TIME_EMB_DIM: usize = 256;
const TIME_EMB_SCALE: f64 = 1000.0;
const ROPE_THETA: f32 = 10000.0;
const LN_EPS: f32 = 1e-6;

/// Mel frames per speech token (pre_la output is repeat_interleave'd by this).
pub const TOKEN_MEL_RATIO: usize = 2;

/// Trim the prompt so ref_mel frames == TOKEN_MEL_RATIO * prompt tokens
/// (CrispASR cv3_synth_with_voice step 0). Returns the prompt-token count to
/// keep; the caller slices `prompt_tokens[..n]` and `ref_mel[..n * TOKEN_MEL_RATIO]`.
pub fn align_prompt_len(n_prompt_tokens: usize, ref_mel_frames: usize) -> usize {
    if ref_mel_frames > 0 {
        (ref_mel_frames / TOKEN_MEL_RATIO).min(n_prompt_tokens)
    } else {
        n_prompt_tokens
    }
}

/// Sinusoidal timestep embedding (cosyvoice DiT SinusPositionEmbedding):
/// first half sin, second half cos, scale 1000, decay ln(10000)/(half-1).
fn sinusoidal_time_emb(t: f64, device: &Device) -> Result<Tensor> {
    let half = TIME_EMB_DIM / 2;
    let decay = 10000f64.ln() / (half - 1) as f64;
    let mut v = vec![0f32; TIME_EMB_DIM];
    for i in 0..half {
        let pos = TIME_EMB_SCALE * t * (-(i as f64) * decay).exp();
        v[i] = pos.sin() as f32;
        v[i + half] = pos.cos() as f32;
    }
    Ok(Tensor::from_vec(v, (1, TIME_EMB_DIM), device)?)
}

/// Seeded standard-normal init noise for the Euler ODE (Box-Muller; any RNG
/// is fine — the ODE output is robust to the noise stream, CrispASR §5478).
fn seeded_randn(n: usize, seed: u64, device: &Device) -> Result<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.random::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        v.push(r * (2.0 * std::f32::consts::PI * u2).cos());
        if v.len() < n {
            v.push(r * (2.0 * std::f32::consts::PI * u2).sin());
        }
    }
    let _ = device;
    Ok(v)
}

/// Cosine t-schedule: t_span[k] = 1 - cos(k/N * pi/2), k in 0..=N.
fn cosine_t_span(n_steps: usize) -> Vec<f64> {
    (0..=n_steps)
        .map(|k| 1.0 - (k as f64 / n_steps as f64 * std::f64::consts::FRAC_PI_2).cos())
        .collect()
}

/// Additive chunk-causal attention mask (1, 1, T, T), F32: entry (i, j) is 0
/// when j < (floor(i/STATIC_CHUNK_SIZE) + 1) * STATIC_CHUNK_SIZE, else -inf
/// (upstream utils/mask.py:154-158 `subsequent_chunk_mask`, applied in
/// DiT/dit.py:163-166 for streaming chunks).
fn chunk_causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut m = vec![f32::NEG_INFINITY; t * t];
    for (i, row) in m.chunks_exact_mut(t).enumerate() {
        let hi = ((i / STATIC_CHUNK_SIZE) + 1) * STATIC_CHUNK_SIZE;
        for v in row.iter_mut().take(hi.min(t)) {
            *v = 0.0;
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, t, t), device)?)
}

/// Upstream RoPE quirk: applied on the PRE-RESHAPE (T, 16*64) Q/K with
/// rot_dim = head_dim = 64, so only the first 64 channels (= head 0) rotate.
/// Interleaved/GPT-J convention (adjacent pairs), theta = 10000.
/// cos/sin: (T, 64) in repeat-interleave format [c0, c0, c1, c1, ...].
fn apply_partial_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rot_dim = cos.dim(D::Minus1)?;
    let d = x.dim(D::Minus1)?;
    let x_rot = x.narrow(D::Minus1, 0, rot_dim)?;
    let x_pass = x.narrow(D::Minus1, rot_dim, d - rot_dim)?;
    let x_rot = x_rot
        .broadcast_mul(cos)?
        .add(&rotate_half_interleave(&x_rot)?.broadcast_mul(sin)?)?;
    Ok(Tensor::cat(&[x_rot, x_pass], D::Minus1)?.contiguous()?)
}

/// One DiT block: AdaLN-Zero modulation (6 x dim from time-emb), bidirectional
/// biased MHA with the partial-RoPE quirk, FFN Linear -> GELU(tanh) -> Linear.
struct DitBlock {
    adaln: Linear, // dim -> 6*dim, chunk order (shift, scale, gate) x {msa, mlp}
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_o: Linear,
    ffn_l1: Linear, // dim -> ff_dim
    ffn_l2: Linear, // ff_dim -> dim
}

impl DitBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            adaln: linear(DIT_DIM, 6 * DIT_DIM, vb.pp("attn_norm.linear"))?,
            to_q: linear(DIT_DIM, DIT_DIM, vb.pp("attn.to_q"))?,
            to_k: linear(DIT_DIM, DIT_DIM, vb.pp("attn.to_k"))?,
            to_v: linear(DIT_DIM, DIT_DIM, vb.pp("attn.to_v"))?,
            to_o: linear(DIT_DIM, DIT_DIM, vb.pp("attn.to_out.0"))?,
            ffn_l1: linear(DIT_DIM, DIT_FF_DIM, vb.pp("ff.ff.0.0"))?,
            ffn_l2: linear(DIT_FF_DIM, DIT_DIM, vb.pp("ff.ff.2"))?,
        })
    }

    /// x: (T, dim); t_silu: (1, dim) precomputed silu(t_emb); cos/sin: (T, 64).
    /// attn_mask: optional additive (1, 1, T, T) chunk-causal mask (0 / -inf).
    fn forward(
        &self,
        x: &Tensor,
        t_silu: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        ln_ones: &Tensor,
        ln_zeros: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t = x.dim(0)?;
        let md = self.adaln.forward(t_silu)?; // (1, 6*dim)
        let chunks = md.chunk(6, 1)?;
        let (shift_msa, scale_msa, gate_msa) = (&chunks[0], &chunks[1], &chunks[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&chunks[3], &chunks[4], &chunks[5]);

        // h = LN(x) * (1 + scale) + shift (affine-free LN, eps 1e-6)
        let modulate = |x: &Tensor, scale: &Tensor, shift: &Tensor| -> Result<Tensor> {
            let lnx = ops::layer_norm(x, ln_ones, ln_zeros, LN_EPS)?;
            Ok(lnx
                .broadcast_mul(&(scale + 1.0)?)?
                .broadcast_add(shift)?
                .contiguous()?)
        };

        // --- MHA ---
        let h = modulate(x, scale_msa, shift_msa)?;
        let q = apply_partial_rope(&self.to_q.forward(&h)?, cos, sin)?;
        let k = apply_partial_rope(&self.to_k.forward(&h)?, cos, sin)?;
        let v = self.to_v.forward(&h)?;
        // (T, dim) -> (1, heads, T, head_dim)
        let to_heads = |y: &Tensor| -> Result<Tensor> {
            Ok(y.reshape((t, DIT_HEADS, DIT_HEAD_DIM))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?)
        };
        // Run the attention math in F32: the UNSCALED q·k matmul overflows
        // F16 (hidden magnitudes reach ~100+, so q·k > 65504 -> inf -> NaN
        // softmax), which is where the F16 path blew up. Casting back after
        // keeps the rest of the estimator in the module dtype; the casts are
        // identity no-ops on the F32 (flow.pt) path.
        let dtype = x.dtype();
        let attn = eager_attention_forward(
            &to_heads(&q)?.to_dtype(DType::F32)?,
            &to_heads(&k)?.to_dtype(DType::F32)?,
            &to_heads(&v)?.to_dtype(DType::F32)?,
            None,
            attn_mask, // None = bidirectional; Some = chunk-causal (streaming)
            1.0 / (DIT_HEAD_DIM as f64).sqrt(),
        )? // (1, T, heads, head_dim)
        .to_dtype(dtype)?;
        let attn = attn.reshape((t, DIT_DIM))?;
        let attn = self.to_o.forward(&attn)?;
        let x = x.add(&attn.broadcast_mul(gate_msa)?)?;

        // --- FFN ---
        let h = modulate(&x, scale_mlp, shift_mlp)?;
        let ff = self.ffn_l2.forward(&self.ffn_l1.forward(&h)?.gelu()?)?; // GELU(tanh)
        Ok(x.add(&ff.broadcast_mul(gate_mlp)?)?)
    }
}

/// CosyVoice3 flow model: token conditioning + DiT-CFM estimator + Euler solver.
pub struct CosyVoice3Flow {
    device: Device,
    /// Module dtype: F32 for the flow.pt path, F16 for the GGUF path.
    dtype: DType,
    // Token conditioning.
    input_embd: Tensor,   // (codebook, mel)
    pre_la_conv1: Conv1d, // k=4, 80 -> 1024, right-pad 3 (lookahead)
    pre_la_conv2: Conv1d, // k=3, 1024 -> 80, left-pad 2 (causal)
    spk_affine: Linear,   // 192 -> 80
    // Estimator input pipeline.
    in_proj: Linear,   // 320 -> 1024
    pos_conv1: Conv1d, // grouped causal conv k=31, groups=16
    pos_conv2: Conv1d,
    time_mlp_0: Linear, // 256 -> 1024
    time_mlp_2: Linear, // 1024 -> 1024
    blocks: Vec<DitBlock>,
    norm_out: Linear, // 1024 -> 2048, chunk order (scale, shift)
    proj_out: Linear, // 1024 -> 80
    rope: RoPE,
    ln_ones: Tensor,  // (dim,) for affine-free LayerNorm
    ln_zeros: Tensor, // (dim,)
}

/// Map one `cosyvoice3.flow.*` GGUF tensor name to the flow.pt module tree.
/// Returns None for tensors we recompute at runtime (rope_inv_freq).
fn flow_gguf_to_pt_name(name: &str) -> Result<Option<String>> {
    let s = name
        .strip_prefix("cosyvoice3.flow.")
        .ok_or_else(|| anyhow!("flow gguf: unexpected tensor name `{name}`"))?;
    let fixed: &[(&str, &str)] = &[
        ("input_embd.w", "input_embedding.weight"),
        ("pre_la.conv1.w", "pre_lookahead_layer.conv1.weight"),
        ("pre_la.conv1.b", "pre_lookahead_layer.conv1.bias"),
        ("pre_la.conv2.w", "pre_lookahead_layer.conv2.weight"),
        ("pre_la.conv2.b", "pre_lookahead_layer.conv2.bias"),
        ("spk_affine.w", "spk_embed_affine_layer.weight"),
        ("spk_affine.b", "spk_embed_affine_layer.bias"),
        ("dit.in_proj.w", "decoder.estimator.input_embed.proj.weight"),
        ("dit.in_proj.b", "decoder.estimator.input_embed.proj.bias"),
        (
            "dit.conv_pos.c1.w",
            "decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight",
        ),
        (
            "dit.conv_pos.c1.b",
            "decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias",
        ),
        (
            "dit.conv_pos.c2.w",
            "decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight",
        ),
        (
            "dit.conv_pos.c2.b",
            "decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias",
        ),
        (
            "dit.time_mlp.0.w",
            "decoder.estimator.time_embed.time_mlp.0.weight",
        ),
        (
            "dit.time_mlp.0.b",
            "decoder.estimator.time_embed.time_mlp.0.bias",
        ),
        (
            "dit.time_mlp.2.w",
            "decoder.estimator.time_embed.time_mlp.2.weight",
        ),
        (
            "dit.time_mlp.2.b",
            "decoder.estimator.time_embed.time_mlp.2.bias",
        ),
        ("dit.norm_out.w", "decoder.estimator.norm_out.linear.weight"),
        ("dit.norm_out.b", "decoder.estimator.norm_out.linear.bias"),
        ("dit.proj_out.w", "decoder.estimator.proj_out.weight"),
        ("dit.proj_out.b", "decoder.estimator.proj_out.bias"),
    ];
    for (from, to) in fixed {
        if s == *from {
            return Ok(Some(to.to_string()));
        }
    }
    if s == "dit.rope_inv_freq" {
        return Ok(None); // recomputed by RoPE::new (theta 10000)
    }
    if let Some(rest) = s.strip_prefix("dit.blk.") {
        let (n, w) = rest
            .split_once('.')
            .ok_or_else(|| anyhow!("flow gguf: bad block tensor name `{name}`"))?;
        let pt_w = match w {
            "adaln.w" => "attn_norm.linear.weight",
            "adaln.b" => "attn_norm.linear.bias",
            "attn.q.w" => "attn.to_q.weight",
            "attn.q.b" => "attn.to_q.bias",
            "attn.k.w" => "attn.to_k.weight",
            "attn.k.b" => "attn.to_k.bias",
            "attn.v.w" => "attn.to_v.weight",
            "attn.v.b" => "attn.to_v.bias",
            "attn.o.w" => "attn.to_out.0.weight",
            "attn.o.b" => "attn.to_out.0.bias",
            "ffn.l1.w" => "ff.ff.0.0.weight",
            "ffn.l1.b" => "ff.ff.0.0.bias",
            "ffn.l2.w" => "ff.ff.2.weight",
            "ffn.l2.b" => "ff.ff.2.bias",
            _ => return Err(anyhow!("flow gguf: unmapped block tensor `{name}`")),
        };
        return Ok(Some(format!(
            "decoder.estimator.transformer_blocks.{n}.{pt_w}"
        )));
    }
    Err(anyhow!("flow gguf: unmapped tensor `{name}`"))
}

/// Read `cosyvoice3-flow-q8_0.gguf`, dequantize every tensor to F16 and key
/// the map by the flow.pt module names (`from_vb` consumes that tree).
fn load_flow_gguf(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut reader = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let ct = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("parse gguf header {}", path.display()))?;
    let mut tensors = HashMap::with_capacity(ct.tensor_infos.len());
    for name in ct.tensor_infos.keys() {
        let Some(pt_name) = flow_gguf_to_pt_name(name)? else {
            continue;
        };
        let t = ct
            .tensor(&mut reader, name, device)
            .with_context(|| format!("gguf tensor `{name}`"))?
            .dequantize(device)?
            .to_dtype(DType::F16)?;
        tensors.insert(pt_name, t);
    }
    if !tensors.contains_key("decoder.estimator.proj_out.weight") {
        bail!(
            "cosyvoice3 flow: {} does not look like a flow GGUF (missing decoder.estimator.*)",
            path.display()
        );
    }
    Ok(tensors)
}

impl CosyVoice3Flow {
    /// Load the flow weights from `dir`: the Q8_0 GGUF (dequantized to F16)
    /// when present, else the upstream `flow.pt` pickle (F32).
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let gguf_path = dir.join("cosyvoice3-flow-q8_0.gguf");
        if gguf_path.exists() {
            let tensors = load_flow_gguf(&gguf_path, device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F16, device);
            return Self::from_vb(vb, device, DType::F16);
        }

        let path = dir.join("flow.pt");
        if !path.exists() {
            bail!("cosyvoice3 flow: {} not found", path.display());
        }
        let dict = match read_all_with_key(&path, Some("state_dict")) {
            Ok(d) => d,
            Err(_) => read_all_with_key(&path, None)
                .map_err(|e| anyhow!("cosyvoice3 flow: read {} failed: {e}", path.display()))?,
        };
        let tensors: HashMap<String, Tensor> = dict.into_iter().collect();
        if !tensors.contains_key("decoder.estimator.proj_out.weight") {
            bail!(
                "cosyvoice3 flow: {} does not look like a flow state dict (missing decoder.estimator.*)",
                path.display()
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Self::from_vb(vb, device, DType::F32)
    }

    /// Build the module tree from a VarBuilder over the flow.pt name space
    /// (the GGUF path remaps its keys to the same tree, F16-dequantized).
    fn from_vb(vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let no_pad = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let grouped = Conv1dConfig {
            groups: 16,
            ..no_pad
        };
        let est = vb.pp("decoder.estimator");
        let mut blocks = Vec::with_capacity(N_DIT_LAYERS);
        for i in 0..N_DIT_LAYERS {
            blocks.push(DitBlock::load(est.pp(&format!("transformer_blocks.{i}")))?);
        }
        Ok(Self {
            device: device.clone(),
            dtype,
            input_embd: vb
                .pp("input_embedding")
                .get((SPEECH_CODEBOOK, MEL_DIM), "weight")?,
            pre_la_conv1: conv1d(
                MEL_DIM,
                DIT_DIM,
                PRE_LOOKAHEAD_LEN + 1,
                no_pad,
                vb.pp("pre_lookahead_layer.conv1"),
            )?,
            pre_la_conv2: conv1d(
                DIT_DIM,
                MEL_DIM,
                PRE_LOOKAHEAD_LEN,
                no_pad,
                vb.pp("pre_lookahead_layer.conv2"),
            )?,
            spk_affine: linear(SPK_DIM_IN, MEL_DIM, vb.pp("spk_embed_affine_layer"))?,
            in_proj: linear(DIT_INPUT_DIM, DIT_DIM, est.pp("input_embed.proj"))?,
            pos_conv1: conv1d(
                DIT_DIM,
                DIT_DIM,
                31,
                grouped,
                est.pp("input_embed.conv_pos_embed.conv1.0"),
            )?,
            pos_conv2: conv1d(
                DIT_DIM,
                DIT_DIM,
                31,
                grouped,
                est.pp("input_embed.conv_pos_embed.conv2.0"),
            )?,
            time_mlp_0: linear(TIME_EMB_DIM, DIT_DIM, est.pp("time_embed.time_mlp.0"))?,
            time_mlp_2: linear(DIT_DIM, DIT_DIM, est.pp("time_embed.time_mlp.2"))?,
            blocks,
            norm_out: linear(DIT_DIM, 2 * DIT_DIM, est.pp("norm_out.linear"))?,
            proj_out: linear(DIT_DIM, MEL_DIM, est.pp("proj_out"))?,
            rope: RoPE::new(DIT_HEAD_DIM, ROPE_THETA, device)?,
            ln_ones: Tensor::ones(DIT_DIM, dtype, device)?,
            ln_zeros: Tensor::zeros(DIT_DIM, dtype, device)?,
        })
    }

    /// Speech tokens -> mu (T_mel = TOKEN_MEL_RATIO * T_tok, mel).
    /// Embedding lookup -> PreLookahead (right-pad 3, conv1 k=4, leaky_relu(0.01),
    /// left-pad 2, conv2 k=3, + residual) -> repeat_interleave(2).
    ///
    /// `context`: the next PRE_LOOKAHEAD_LEN tokens when streaming
    /// (finalize=false): their embeddings replace the zero right-pad before
    /// conv1 (upstream PreLookaheadLayer main/context split,
    /// transformer/upsample_encoder.py:91-94 via flow/flow.py:394).
    fn pre_lookahead_mu(&self, speech_tokens: &[u32], context: Option<&[u32]>) -> Result<Tensor> {
        let t_tok = speech_tokens.len();
        if speech_tokens.iter().any(|&t| t as usize >= SPEECH_CODEBOOK) {
            bail!("cosyvoice3 flow: speech token id out of range (codebook {SPEECH_CODEBOOK})");
        }
        let ids = Tensor::from_vec(speech_tokens.to_vec(), t_tok, &self.device)?;
        let tok_emb = self.input_embd.index_select(&ids, 0)?; // (T_tok, mel)
        // Conv chain is channel-first: (1, C, T).
        let x = tok_emb.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // (1, mel, T)
        let x = match context {
            None => x.pad_with_zeros(2, 0, PRE_LOOKAHEAD_LEN)?, // right-pad 3
            Some(ctx) => {
                if ctx.len() != PRE_LOOKAHEAD_LEN {
                    bail!(
                        "cosyvoice3 flow: pre-lookahead context must be {PRE_LOOKAHEAD_LEN} tokens, got {}",
                        ctx.len()
                    );
                }
                if ctx.iter().any(|&t| t as usize >= SPEECH_CODEBOOK) {
                    bail!(
                        "cosyvoice3 flow: context token id out of range (codebook {SPEECH_CODEBOOK})"
                    );
                }
                let cids = Tensor::from_vec(ctx.to_vec(), ctx.len(), &self.device)?;
                let ctx_emb = self.input_embd.index_select(&cids, 0)?; // (3, mel)
                let ctx = ctx_emb.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // (1, mel, 3)
                // context fills the whole lookahead window -> no zero pad left
                Tensor::cat(&[&x, &ctx], 2)?
            }
        };
        let x = self.pre_la_conv1.forward(&x)?; // (1, 1024, T)
        let x = ops::leaky_relu(&x, 0.01)?;
        let x = x.pad_with_zeros(2, PRE_LOOKAHEAD_LEN - 1, 0)?; // left-pad 2
        let x = self.pre_la_conv2.forward(&x)?; // (1, mel, T)
        let x = x.squeeze(0)?.transpose(0, 1)?.contiguous()?; // (T_tok, mel)
        let pre_la = x.add(&tok_emb)?; // residual
        // repeat_interleave(TOKEN_MEL_RATIO) along time.
        Ok(pre_la
            .unsqueeze(1)?
            .repeat((1, TOKEN_MEL_RATIO, 1))?
            .reshape((t_tok * TOKEN_MEL_RATIO, MEL_DIM))?)
    }

    /// F.normalize(spk_emb) -> spk_affine. Accepts (192,) or (1, 192).
    fn spk_project(&self, spk_emb: &Tensor) -> Result<Tensor> {
        let spk = spk_emb
            .to_device(&self.device)?
            .to_dtype(DType::F32)?
            .flatten_all()?;
        if spk.dim(0)? != SPK_DIM_IN {
            bail!(
                "cosyvoice3 flow: spk_emb dim {} != {SPK_DIM_IN}",
                spk.dim(0)?
            );
        }
        // F.normalize: x / max(||x||_2, 1e-12); the scalar sync is once per call.
        let norm = spk.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-12);
        let spk = (spk / norm as f64)?;
        let spk = spk.to_dtype(self.dtype)?.unsqueeze(0)?;
        Ok(self.spk_affine.forward(&spk)?) // (1, mel)
    }

    /// DiT estimator velocity prediction. x/mu/cond: (T_mel, mel);
    /// spk_proj: (1, mel); cos/sin: (T_mel, 64). Returns (T_mel, mel).
    /// `attn_mask`: the chunk-causal mask for streaming chunks (built once
    /// per chunk by the caller; upstream DiT/dit.py:163-166 +
    /// utils/mask.py:154-158 `subsequent_chunk_mask(., static_chunk_size)`),
    /// None = bidirectional (finalize).
    #[allow(clippy::too_many_arguments)]
    fn estimator_forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        spk_proj: &Tensor,
        cond: &Tensor,
        t: f64,
        cos: &Tensor,
        sin: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t_mel = x.dim(0)?;
        // TimestepEmbedding: sinusoidal -> Linear -> SiLU -> Linear.
        let sin_emb = sinusoidal_time_emb(t, &self.device)?.to_dtype(self.dtype)?;
        let t_emb = self
            .time_mlp_2
            .forward(&ops::silu(&self.time_mlp_0.forward(&sin_emb)?)?)?; // (1, dim)
        let t_silu = ops::silu(&t_emb)?;

        // InputEmbedding: cat[x, cond, mu, spks_broadcast] -> Linear -> conv_pos + residual.
        let spk_bc = spk_proj.broadcast_as((t_mel, MEL_DIM))?.contiguous()?;
        let catted = Tensor::cat(&[x, cond, mu, &spk_bc], 1)?; // (T, 320)
        let proj = self.in_proj.forward(&catted)?; // (T, dim)
        let pos = proj.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // (1, dim, T)
        let pos = pos.pad_with_zeros(2, 30, 0)?; // causal left-pad k-1
        let pos = ops::mish(&self.pos_conv1.forward(&pos)?)?;
        let pos = pos.pad_with_zeros(2, 30, 0)?;
        let pos = ops::mish(&self.pos_conv2.forward(&pos)?)?;
        let pos = pos.squeeze(0)?.transpose(0, 1)?.contiguous()?; // (T, dim)
        let mut h = proj.add(&pos)?;

        for blk in &self.blocks {
            h = blk.forward(
                &h,
                &t_silu,
                cos,
                sin,
                &self.ln_ones,
                &self.ln_zeros,
                attn_mask,
            )?;
        }

        // AdaLN-Final: Linear(silu(t_emb)) -> 2*dim in (scale, shift) order.
        let nmod = self.norm_out.forward(&t_silu)?; // (1, 2*dim)
        let chunks = nmod.chunk(2, 1)?;
        let (nscale, nshift) = (&chunks[0], &chunks[1]);
        let lnx = ops::layer_norm(&h, &self.ln_ones, &self.ln_zeros, LN_EPS)?;
        let normed = lnx.broadcast_mul(&(nscale + 1.0)?)?.broadcast_add(nshift)?;
        Ok(self.proj_out.forward(&normed)?) // (T, mel)
    }

    /// Cosine-schedule Euler ODE with classifier-free guidance.
    /// Runs the estimator twice per step (cond / uncond) — see module header.
    /// Interval-CFG (env `CV3_CFG_INTERVAL=K`, 0/absent = off): the uncond
    /// branch is recomputed only on the first/last/every-K-th step and cached
    /// in between (CrispASR's `CRISPASR_COSYVOICE3_CFG_INTERVAL` semantics) —
    /// near-lossless, cuts most of the uncond cost.
    /// `finalize == false` puts the estimator in streaming-chunk mode
    /// (chunk-causal attention mask on both CFG branches, as upstream runs
    /// cond+uncond batched through the same masked estimator).
    fn solve_euler(
        &self,
        mu: &Tensor,
        spk_proj: &Tensor,
        cond: &Tensor,
        n_steps: usize,
        cfg_rate: f64,
        seed: u64,
        finalize: bool,
    ) -> Result<Tensor> {
        let (t_mel, _) = mu.dims2()?;
        let t_span = cosine_t_span(n_steps);
        let noise = seeded_randn(t_mel * MEL_DIM, seed, &self.device)?;
        let mut x =
            Tensor::from_vec(noise, (t_mel, MEL_DIM), &self.device)?.to_dtype(self.dtype)?;
        let (cos, sin) = self
            .rope
            .forward_repeat_interleave(0, t_mel, &self.device)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;
        // Streaming chunk: one chunk-causal mask for the whole ODE (depends
        // only on t_mel; mask[i, j] = j < (floor(i/50) + 1) * 50).
        let attn_mask = if finalize {
            None
        } else {
            Some(chunk_causal_mask(t_mel, &self.device)?)
        };
        let attn_mask = attn_mask.as_ref();
        let zeros_mel = Tensor::zeros((t_mel, MEL_DIM), self.dtype, &self.device)?;
        let zeros_spk = Tensor::zeros((1, MEL_DIM), self.dtype, &self.device)?;
        let cfg_interval: usize = std::env::var("CV3_CFG_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut cached_unc: Option<Tensor> = None;
        let mut t = t_span[0];
        let mut dt = t_span[1] - t_span[0];
        for step in 1..=n_steps {
            let dphi_cond =
                self.estimator_forward(&x, mu, spk_proj, cond, t, &cos, &sin, attn_mask)?;
            let dphi = if cfg_rate != 0.0 {
                // Uncond branch: mu / spks / cond all zeroed. With interval-CFG,
                // reuse the cached dphi from a nearby step (approximation).
                let recompute = cached_unc.is_none()
                    || step == n_steps
                    || (cfg_interval > 0 && step % cfg_interval == 0);
                if recompute {
                    cached_unc = Some(self.estimator_forward(
                        &x, &zeros_mel, &zeros_spk, &zeros_mel, t, &cos, &sin, attn_mask,
                    )?);
                }
                let dphi_unc = cached_unc.as_ref().unwrap();
                ((dphi_cond * (1.0 + cfg_rate))? - (dphi_unc * cfg_rate)?)?
            } else {
                dphi_cond
            };
            x = (x + (dphi * dt)?)?;
            t += dt;
            if step < n_steps {
                dt = t_span[step + 1] - t;
            }
        }
        Ok(x)
    }

    /// Full flow synthesis: speech tokens (prompt prefix + generated, already
    /// alignment-trimmed via `align_prompt_len`), speaker embedding (192) and
    /// ref mel (T_ref = TOKEN_MEL_RATIO * n_prompt_tokens, 80) -> generated mel
    /// (T_mel, 80) INCLUDING the ref prefix frames (pipeline slices them off).
    pub fn synthesize_mel(
        &self,
        speech_tokens: &[u32],
        spk_emb: &Tensor,
        ref_mel: &Tensor,
        n_steps: usize,
        cfg: f64,
        seed: u64,
    ) -> Result<Tensor> {
        self.synthesize_mel_impl(
            speech_tokens,
            None,
            spk_emb,
            ref_mel,
            n_steps,
            cfg,
            seed,
            true,
        )
    }

    /// Streaming-chunk variant (upstream flow/flow.py:391-394).
    /// `full_tokens` = prompt prefix + generated tokens SO FAR + the
    /// PRE_LOOKAHEAD_LEN lookahead tokens (the last 3).
    /// finalize=true -> identical to `synthesize_mel` over all tokens.
    /// finalize=false -> the last 3 tokens are split off as PreLookahead
    /// context (not decoded into mel), and the DiT runs with the
    /// chunk-causal attention mask. The same `seed` keeps x_init a
    /// deterministic prefix across chunks (seeded_randn draws sequentially).
    /// Returns mel INCLUDING the ref prefix frames.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_mel_chunk(
        &self,
        full_tokens: &[u32],
        spk_emb: &Tensor,
        ref_mel: &Tensor,
        n_steps: usize,
        cfg: f64,
        seed: u64,
        finalize: bool,
    ) -> Result<Tensor> {
        if finalize {
            return self.synthesize_mel(full_tokens, spk_emb, ref_mel, n_steps, cfg, seed);
        }
        if full_tokens.len() <= PRE_LOOKAHEAD_LEN {
            bail!(
                "cosyvoice3 flow: streaming chunk needs > {PRE_LOOKAHEAD_LEN} tokens, got {}",
                full_tokens.len()
            );
        }
        let (main, context) = full_tokens.split_at(full_tokens.len() - PRE_LOOKAHEAD_LEN);
        self.synthesize_mel_impl(
            main,
            Some(context),
            spk_emb,
            ref_mel,
            n_steps,
            cfg,
            seed,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_mel_impl(
        &self,
        speech_tokens: &[u32],
        context: Option<&[u32]>,
        spk_emb: &Tensor,
        ref_mel: &Tensor,
        n_steps: usize,
        cfg: f64,
        seed: u64,
        finalize: bool,
    ) -> Result<Tensor> {
        if speech_tokens.is_empty() {
            bail!("cosyvoice3 flow: empty speech token sequence");
        }
        if n_steps == 0 {
            bail!("cosyvoice3 flow: n_steps must be >= 1");
        }
        let mu = self.pre_lookahead_mu(speech_tokens, context)?; // (T_mel, mel)
        let (t_mel, _) = mu.dims2()?;
        let spk_proj = self.spk_project(spk_emb)?; // (1, mel)

        // cond: ref_mel in the first T_ref frames, zeros after.
        let ref_mel = ref_mel.to_device(&self.device)?.to_dtype(self.dtype)?;
        let (t_ref, mel) = ref_mel.dims2()?;
        if mel != MEL_DIM {
            bail!("cosyvoice3 flow: ref_mel dim {mel} != {MEL_DIM}");
        }
        if t_ref > t_mel {
            bail!("cosyvoice3 flow: ref_mel ({t_ref}) longer than full mel ({t_mel})");
        }
        let cond = if t_ref == t_mel {
            ref_mel
        } else {
            Tensor::cat(
                &[
                    &ref_mel,
                    &Tensor::zeros((t_mel - t_ref, MEL_DIM), self.dtype, &self.device)?,
                ],
                0,
            )?
        };

        self.solve_euler(&mu, &spk_proj, &cond, n_steps, cfg, seed, finalize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_prompt_len() {
        // Baked zero_shot voice: already aligned (174 == 2*87) — no-op.
        assert_eq!(align_prompt_len(87, 174), 87);
        // ref_mel shorter than 2*tokens: trim tokens.
        assert_eq!(align_prompt_len(100, 174), 87);
        // No ref mel: keep all tokens.
        assert_eq!(align_prompt_len(100, 0), 100);
        // ref_mel longer than 2*tokens: keep all tokens.
        assert_eq!(align_prompt_len(10, 1000), 10);
    }

    #[test]
    fn test_chunk_causal_mask() {
        // T=120: rows 0..50 see j<50, rows 50..100 see j<100, rows 100..120 all.
        let m = chunk_causal_mask(120, &Device::Cpu)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(m[0][..50].iter().all(|&v| v == 0.0));
        assert!(m[0][50..].iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(m[49][..50].iter().all(|&v| v == 0.0));
        assert!(m[50][..100].iter().all(|&v| v == 0.0));
        assert_eq!(m[50][100], f32::NEG_INFINITY);
        assert!(m[119].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_cosine_t_span() {
        let ts = cosine_t_span(10);
        assert_eq!(ts.len(), 11);
        assert!((ts[0] - 0.0).abs() < 1e-12);
        assert!((ts[10] - 1.0).abs() < 1e-12);
        // Strictly increasing, first step smallest (cosine ramp).
        for w in ts.windows(2) {
            assert!(w[1] > w[0]);
        }
        assert!((ts[5] - (1.0 - (0.5 * std::f64::consts::FRAC_PI_2).cos())).abs() < 1e-12);
    }

    #[test]
    fn test_seeded_randn_deterministic() {
        let dev = Device::Cpu;
        let a = seeded_randn(64, 42, &dev).unwrap();
        let b = seeded_randn(64, 42, &dev).unwrap();
        let c = seeded_randn(64, 43, &dev).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Roughly standard normal.
        let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn test_sinusoidal_time_emb() {
        let emb = sinusoidal_time_emb(0.5, &Device::Cpu).unwrap();
        assert_eq!(emb.dims2().unwrap(), (1, TIME_EMB_DIM));
        let v = emb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // t=0.5, scale=1000, i=0: freq=1 -> sin(500), cos(500).
        assert!((v[0] - 500f64.sin() as f32).abs() < 1e-4);
        assert!((v[TIME_EMB_DIM / 2] - 500f64.cos() as f32).abs() < 1e-4);
        // t=0 -> sin=0, cos=1.
        let e0 = sinusoidal_time_emb(0.0, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(e0[..TIME_EMB_DIM / 2].iter().all(|&x| x == 0.0));
        assert!(e0[TIME_EMB_DIM / 2..].iter().all(|&x| x == 1.0));
    }

    /// Runtime smoke test against the real weights: loads flow.pt and runs a
    /// tiny end-to-end synthesis on CPU. Run explicitly with:
    ///   cargo test cosyvoice3::flow -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_synthesize_mel_smoke() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/Fun-CosyVoice3-0.5B-2512");
        if !dir.join("flow.pt").exists() {
            eprintln!("flow.pt not downloaded; skipping");
            return;
        }
        let dev = Device::Cpu;
        let flow = CosyVoice3Flow::load(&dir, &dev).unwrap();
        // 2 prompt tokens (4 ref frames) + 6 generated tokens.
        let tokens: Vec<u32> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let spk = Tensor::randn(0f32, 1f32, (SPK_DIM_IN,), &dev).unwrap();
        let ref_mel = Tensor::randn(0f32, 1f32, (4, MEL_DIM), &dev).unwrap();
        let mel = flow
            .synthesize_mel(&tokens, &spk, &ref_mel, 2, 0.7, 0)
            .unwrap();
        assert_eq!(
            mel.dims2().unwrap(),
            (tokens.len() * TOKEN_MEL_RATIO, MEL_DIM)
        );
        // F32 on the flow.pt path, F16 on the GGUF path.
        assert!(matches!(mel.dtype(), DType::F32 | DType::F16));
        // Output must be finite (no NaN/Inf blow-ups in the ODE).
        let flat = mel
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
        eprintln!(
            "smoke mel: {:?} absmax {:.3}",
            mel.shape(),
            flat.iter().fold(0f32, |a, &b| a.max(b.abs()))
        );
    }
}
