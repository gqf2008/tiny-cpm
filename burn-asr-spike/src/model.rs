//! Qwen3-ASR model on burn 0.22.0-pre.1 (Metal backend): whisper-style audio
//! tower + Qwen3 text decoder with mrope + KV cache + tied lm_head.
//!
//! Ported 1:1 from tiny-cpm (candle) src/models/qwen3_asr/model.rs,
//! src/models/qwen3/model.rs, src/common/modules.rs (NaiveAttention /
//! QKNormAttention / GateUpDownMLP / eager_attention_forward),
//! src/position_embed/{rope,sinusoidal}.rs and src/utils/tensor_utils.rs.
//! Weight names and tensor shapes are identical to the safetensors file.

use anyhow::{Result, anyhow};
use burn::tensor::{
    activation, ops::ConvOptions, module, Bool, DType, Device, Int, Tensor,
};

const NEG_INF: f32 = f32::NEG_INFINITY;

/// dtype used for all model tensors (f16 — cubecl-wgpu Metal has no BF16).
/// BURN_ASR_F32=1 overrides to f32 (diagnostic for f16 numeric issues).
pub fn dt() -> DType {
    if std::env::var("BURN_ASR_F32").is_ok() {
        DType::F32
    } else {
        DType::F16
    }
}

fn linear(x: Tensor<3>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<3> {
    let w2 = w.swap_dims(0, 1).unsqueeze::<3>();
    let out = x.matmul(w2);
    match b {
        Some(b) => out + b.unsqueeze::<2>().unsqueeze::<3>(),
        None => out,
    }
}

/// candle RmsNorm: x * rsqrt(mean(x^2) + eps) * w, over the last dim (kept).
///
/// f16 gotcha: this model's bf16 activations legitimately reach ~5800, so
/// x^2 (~3.4e7) overflows f16 (max 65504) and the variance becomes inf →
/// rsqrt = 0 → the whole norm collapses to zero → layers become identity
/// skips. The variance is therefore computed in f32 (inputs are f16-safe;
/// only the square overflows). bf16 in candle doesn't need this (8-bit
/// exponent).
fn rms_norm(x: Tensor<3>, w: Tensor<1>, eps: f64) -> Tensor<3> {
    let dt = x.dtype();
    let xf = x.cast(DType::F32);
    let var = xf.clone().powf_scalar(2.0).mean_dim(2);
    let y = xf * (var + eps).powf_scalar(-0.5);
    y.cast(dt) * w.unsqueeze::<2>().unsqueeze::<3>()
}

/// candle LayerNorm with remove_mean=true. Variance in f32 (see rms_norm).
fn layer_norm(x: Tensor<3>, w: Tensor<1>, b: Tensor<1>, eps: f64) -> Tensor<3> {
    let dt = x.dtype();
    let xf = x.cast(DType::F32);
    let centered = xf.clone() - xf.clone().mean_dim(2);
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    let y = centered * (var + eps).powf_scalar(-0.5);
    y.cast(dt) * w.unsqueeze::<2>().unsqueeze::<3>() + b.unsqueeze::<2>().unsqueeze::<3>()
}

/// rms_norm over the last dim of a rank-4 tensor (q/k norm after reshape).
fn rms_norm4(x: Tensor<4>, w: Tensor<1>, eps: f64) -> Tensor<4> {
    let var = x.clone().powf_scalar(2.0).mean_dim(3);
    x * (var + eps).powf_scalar(-0.5) * w.unsqueeze::<2>().unsqueeze::<3>().unsqueeze()
}

/// GPT-NeoX split-half rotation (candle rotate_half + apply_rotary_pos_emb).
/// cos/sin: (b, 1, s, dim); x: (b, h, s, dim).
fn apply_rope(x: Tensor<4>, cos: Tensor<3>, sin: Tensor<3>) -> Tensor<4> {
    let cos = cos.unsqueeze_dim::<4>(1);
    let sin = sin.unsqueeze_dim::<4>(1);
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rot = Tensor::cat(vec![x2.neg(), x1], 3);
    x * cos + rot * sin
}

/// Repeat each kv head n_rep times (candle repeat_kv). Must match candle's
/// layout exactly: cat along the SEQ dim then reshape, which interleaves the
/// copies ([h0,h0,h1,h1,...]) so q head i attends kv head i/n_rep. Catting
/// along the head dim instead would pair q heads >= n_kv_heads with the wrong
/// kv heads and babble.
fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    let [b, h, s, dd] = x.dims();
    Tensor::cat(vec![x; n_rep], 2).reshape([b, h * n_rep, s, dd])
}

/// Eager attention: q (b,h,s,dd) × k/v (b,h,s_kv,dd) + mask + softmax.
fn eager_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, mask: Option<Tensor<4>>, scale: f64) -> Tensor<4> {
    let mut attn = q.matmul(k.swap_dims(2, 3)) * scale;
    if let Some(m) = mask {
        attn = attn + m;
    }
    let attn = activation::softmax(attn, 3);
    attn.matmul(v)
}

// ---------------------------------------------------------------------------
// Audio tower (whisper-style encoder)
// ---------------------------------------------------------------------------

struct AudioEncoderLayer {
    q_w: Tensor<2>, q_b: Tensor<1>, k_w: Tensor<2>, k_b: Tensor<1>,
    v_w: Tensor<2>, v_b: Tensor<1>, o_w: Tensor<2>, o_b: Tensor<1>,
    attn_ln_w: Tensor<1>, attn_ln_b: Tensor<1>,
    fc1_w: Tensor<2>, fc1_b: Tensor<1>, fc2_w: Tensor<2>, fc2_b: Tensor<1>,
    final_ln_w: Tensor<1>, final_ln_b: Tensor<1>,
    d_model: usize,
    heads: usize,
    head_dim: usize,
}

impl AudioEncoderLayer {
    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let residual = xs.clone();
        let h = layer_norm(xs, self.attn_ln_w.clone(), self.attn_ln_b.clone(), 1e-5);
        let h = self.attn(h, None);
        let h = h + residual;
        let residual = h.clone();
        let h = layer_norm(h, self.final_ln_w.clone(), self.final_ln_b.clone(), 1e-5);
        let h = linear(h, self.fc1_w.clone(), Some(self.fc1_b.clone()));
        let h = activation::gelu(h);
        let h = linear(h, self.fc2_w.clone(), Some(self.fc2_b.clone()));
        h + residual
    }

    /// NaiveAttention (bias, no RoPE, no kv cache). mask unused here (None).
    fn attn(&self, xs: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        let [b, s, _] = xs.dims();
        let q = linear(xs.clone(), self.q_w.clone(), Some(self.q_b.clone()))
            .reshape([b, s, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let k = linear(xs.clone(), self.k_w.clone(), Some(self.k_b.clone()))
            .reshape([b, s, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let v = linear(xs, self.v_w.clone(), Some(self.v_b.clone()))
            .reshape([b, s, self.heads, self.head_dim])
            .swap_dims(1, 2);
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let out = eager_attention(q, k, v, mask, scale);
        let out = out.swap_dims(1, 2).reshape([b, s, self.heads * self.head_dim]);
        linear(out, self.o_w.clone(), Some(self.o_b.clone()))
    }
}

struct AudioEncoder {
    device: Device,
    n_window: usize,
    conv_chunksize: usize,
    conv2d1_w: Tensor<4>, conv2d1_b: Tensor<1>,
    conv2d2_w: Tensor<4>, conv2d2_b: Tensor<1>,
    conv2d3_w: Tensor<4>, conv2d3_b: Tensor<1>,
    conv_out_w: Tensor<2>,
    proj1_w: Tensor<2>, proj1_b: Tensor<1>,
    proj2_w: Tensor<2>, proj2_b: Tensor<1>,
    ln_post_w: Tensor<1>, ln_post_b: Tensor<1>,
    layers: Vec<AudioEncoderLayer>,
    inv_freq: Vec<f32>,
}

impl AudioEncoder {
    /// xs: (num_mel_bins, feature_len) f16 — candle's (feature_dim, feature_len).
    fn forward(&self, xs: Tensor<2>) -> Result<Tensor<2>> {
        let frames = xs.dims()[1];
        let chunk_len = self.n_window * 2; // 100
        let chunk_num = frames / chunk_len;
        let mut chunk_lengths = vec![chunk_len; chunk_num];
        let last = frames % chunk_len;
        if last > 0 {
            chunk_lengths.push(last);
        }
        // split along the frame dim: (mel, frames) -> (mel, chunk) pieces
        let t = xs.swap_dims(0, 1); // (frames, mel)
        let mut chunks = t.split_with_sizes(chunk_lengths.clone(), 0);
        if last > 0 {
            let c = chunks.pop().unwrap();
            let pad = chunk_len - last;
            let mel = c.dims()[1];
            let zeros = Tensor::<2, burn::tensor::Float>::zeros([pad, mel], &self.device).cast(dt());
            chunks.push(Tensor::cat(vec![c, zeros], 0));
        }
        let padded: Tensor<3> = Tensor::stack(chunks, 0); // (n, 100, mel) — stack adds the window dim
        let padded = padded.swap_dims(1, 2).unsqueeze_dim::<4>(1); // (n, 1, mel, 100)

        // convs, in batches of conv_chunksize along dim 0
        let n = padded.dims()[0];
        let mut embeds = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + self.conv_chunksize).min(n);
            let chunk = padded.clone().narrow(0, start, end - start);
            let e = module::conv2d(
                chunk,
                self.conv2d1_w.clone(),
                Some(self.conv2d1_b.clone()),
                conv_opts(),
            );
            let e = activation::gelu(e);
            let e = module::conv2d(e, self.conv2d2_w.clone(), Some(self.conv2d2_b.clone()), conv_opts());
            let e = activation::gelu(e);
            let e = module::conv2d(e, self.conv2d3_w.clone(), Some(self.conv2d3_b.clone()), conv_opts());
            let e = activation::gelu(e);
            embeds.push(e);
            start = end;
        }
        let e = Tensor::cat(embeds, 0); // (n, 480, 16, 13)
        let [b, _, f, t] = e.dims();
        // candle permute((0, 3, 1, 2)): (n, 13, 480, 16) — swap_dims(1,3) alone
        // would give (n, 13, 16, 480) and scramble the reshape order.
        let e = e.permute([0, 3, 1, 2]).reshape([b, t, 480 * f]); // (n, 13, 7680)
        let e = linear(e, self.conv_out_w.clone(), None); // (n, 13, d_model)

        // sinusoidal additive position embedding, per-chunk positions 0..12:
        // candle's SinusoidalPositionEncoderCat reads dim(1) of the (n, 13, d)
        // tensor, so EVERY chunk gets the same 13 positions, then flatten.
        let [n2, s2, d2] = e.dims();
        let pos = self.pos_embed(s2, d2).unsqueeze::<3>(); // (1, 13, d)
        let e = (e + pos).reshape([n2 * s2, d2]);
        let e = e.narrow(0, 0, feat_len_after(&chunk_lengths)).unsqueeze::<3>(); // (1, T', d)

        let mut hidden_states = e;
        for layer in self.layers.iter() {
            hidden_states = layer.forward(hidden_states);
        }
        let e = layer_norm(hidden_states, self.ln_post_w.clone(), self.ln_post_b.clone(), 1e-5);
        let e = linear(e, self.proj1_w.clone(), Some(self.proj1_b.clone()));
        let e = activation::gelu(e);
        let e = linear(e, self.proj2_w.clone(), Some(self.proj2_b.clone())); // (1, T', output_dim)
        Ok(e.squeeze_dim::<2>(0))
    }

    /// SinusoidalPositionEncoderCat: cat([sin, cos]) of positions × inv_freq.
    fn pos_embed(&self, seq_len: usize, dim: usize) -> Tensor<2> {
        let half = dim / 2;
        let pos = Tensor::arange(0..seq_len as i64, &self.device)
            .float()
            .reshape([seq_len, 1]);
        let inv = Tensor::<1>::from_floats(self.inv_freq.as_slice(), &self.device).reshape([1, half]);
        let freqs = pos.matmul(inv); // (seq_len, half)
        let sin = freqs.clone().sin();
        let cos = freqs.cos();
        let e: Tensor<2> = Tensor::cat(vec![sin, cos], 1).cast(dt());
        e
    }
}

fn conv_opts() -> ConvOptions<2> {
    ConvOptions::new([2, 2], [1, 1], [1, 1], 1)
}

fn feat_len_after(chunk_lengths: &[usize]) -> usize {
    chunk_lengths
        .iter()
        .map(|&i| crate::audio::get_feat_extract_output_lengths(i))
        .sum()
}

// ---------------------------------------------------------------------------
// Qwen3 text decoder
// ---------------------------------------------------------------------------

struct DecoderLayer {
    q_w: Tensor<2>, k_w: Tensor<2>, v_w: Tensor<2>, o_w: Tensor<2>,
    q_norm_w: Tensor<1>, k_norm_w: Tensor<1>,
    gate_w: Tensor<2>, up_w: Tensor<2>, down_w: Tensor<2>,
    in_ln_w: Tensor<1>, post_ln_w: Tensor<1>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    kv_cache: Option<(Tensor<4>, Tensor<4>)>,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: Tensor<3>,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
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
        cos: &Tensor<3>,
        sin: &Tensor<3>,
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
        let out = out.swap_dims(1, 2).reshape([b, s, self.n_heads * self.head_dim]);
        linear(out, self.o_w.clone(), None)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// mrope cos/sin: (b, seq, dim). position_ids (3, b, seq) int.
fn rotary_emb(
    device: &Device,
    inv_freq: &[f32],
    position_ids: Tensor<3, Int>,
    mrope_section: &[usize],
) -> (Tensor<3>, Tensor<3>) {
    let [three, b, seq] = position_ids.dims();
    debug_assert_eq!(three, 3);
    let half = inv_freq.len();
    let pos = position_ids.float().unsqueeze_dim::<4>(2); // (3, b, 1, seq)
    let inv = Tensor::<1>::from_floats(inv_freq, device)
        .reshape([1, 1, half, 1])
        .repeat_dim(0, 3)
        .repeat_dim(1, b); // (3, b, half, 1)
    let freqs = inv.matmul(pos).swap_dims(2, 3); // (3, b, seq, half)

    // apply_interleaved_mrope_asr: rows 1..3 scattered into row 0. In the ASR
    // path position_ids is a broadcast arange (all three rows identical), so
    // the interleave is a numeric no-op — and burn 0.22's Float `scatter`
    // implements only IndexingUpdateOp::Add (bridge/ops/float.rs:203), so the
    // Assign form candle uses would panic anyway. Skipped by construction.
    let freqs_t = freqs.narrow(0, 0, 1).squeeze_dim::<3>(0); // (b, seq, half)
    let emb = Tensor::cat(vec![freqs_t.clone(), freqs_t], 2); // (b, seq, dim)
    let cos: Tensor<3> = emb.clone().cos().cast(dt());
    let sin: Tensor<3> = emb.sin().cast(dt());
    (cos, sin)
}

/// (b, 1, s, s) f16 causal mask, -inf strictly above the diagonal.
///
/// Gotcha: burn's triu_mask/tril_mask naming is inverted vs torch — triu_mask
/// marks the *lower* triangle true (doc example: triu_mask([3,3], 0) is true at
/// j < i). tril_mask([s,s], 0) is the one that is true strictly above the
/// diagonal (j > i), which is what a causal mask must set to -inf.
fn causal_mask(device: &Device, b: usize, s: usize) -> Tensor<4> {
    let upper = Tensor::<2, Bool>::tril_mask([s, s], 0, device);
    let mask = Tensor::<2, burn::tensor::Float>::full([s, s], 0.0_f32, device)
        .mask_fill(upper, NEG_INF);
    mask.unsqueeze::<3>().unsqueeze::<4>().repeat_dim(0, b).cast(dt())
}

struct TextModel {
    embed_tokens: Tensor<2>,
    layers: Vec<DecoderLayer>,
    norm_w: Tensor<1>,
    inv_freq: Vec<f32>,
    mrope_section: Vec<usize>,
    eps: f64,
    device: Device,
}

impl TextModel {
    fn forward(&mut self, input_embeds: Tensor<3>, seqlen_offset: usize) -> Tensor<3> {
        let [b, seq, _] = input_embeds.dims();
        // position_ids (3, b, seq): all rows identical (candle arange broadcast)
        let pos = Tensor::arange(
            seqlen_offset as i64..(seqlen_offset + seq) as i64,
            &self.device,
        )
        .reshape([1, 1, seq])
        .repeat_dim(0, 3)
        .repeat_dim(1, b);
        let (cos, sin) = rotary_emb(&self.device, &self.inv_freq, pos, &self.mrope_section);
        let mask = if seq <= 1 {
            None
        } else {
            Some(causal_mask(&self.device, b, seq))
        };
        let mut xs = input_embeds;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(xs, &cos, &sin, mask.as_ref());
        }
        rms_norm(xs, self.norm_w.clone(), self.eps)
    }

    fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

pub struct Qwen3AsrModel {
    audio: AudioEncoder,
    text: TextModel,
    audio_token_id: u32,
    device: Device,
}

impl Qwen3AsrModel {
    /// forward one step. input_ids (1, seq) int; input_features Some only on
    /// the first step. Returns logits (vocab,).
    pub fn forward(
        &mut self,
        input_ids: Tensor<2, Int>,
        seqlen_offset: usize,
        input_features: Option<Tensor<2>>,
    ) -> Result<Tensor<1>> {
        let [_, seq] = input_ids.dims();
        let mut emb = module::embedding(self.text.embed_tokens.clone(), input_ids.clone());
        if let Some(feats) = input_features {
            let audio_feature = self.audio.forward(feats)?; // (T', output_dim)
            let mask = input_ids.equal_scalar(self.audio_token_id as i32); // (1, seq) bool
            emb = masked_scatter_dim0(emb, audio_feature, mask)?;
        }
        let hidden = self.text.forward(emb, seqlen_offset); // (1, seq, D)
        let last = hidden.narrow(1, seq - 1, 1); // (1, 1, D)
        let logits = last.matmul(self.text.embed_tokens.clone().swap_dims(0, 1).unsqueeze::<3>()); // (1, 1, V)
        let v = logits.dims()[2];
        Ok(logits.reshape([v]))
    }

    pub fn clear_cache(&mut self) {
        self.text.clear_cache();
    }
}

/// masked_scatter_dim0: replace the rows of `original` at mask positions with
/// `replace` rows, in order. bs must be 1. (one device sync on the mask, same
/// as the candle version's to_scalar.)
///
/// Note: burn's Float `select_assign` only implements IndexingUpdateOp::Add
/// (bridge/ops/float.rs), so this uses candle's original contiguous-run +
/// `slice_assign` formulation (slice_assign is fully implemented).
fn masked_scatter_dim0(
    original: Tensor<3>,
    replace: Tensor<2>,
    mask: Tensor<2, Bool>,
) -> Result<Tensor<3>> {
    let m = mask.to_data();
    let flags: Vec<bool> = m.iter::<bool>().collect();
    let n = flags.iter().filter(|&&b| b).count();
    let feat_len = replace.dims()[0];
    if n != feat_len {
        return Err(anyhow!(
            "n_audio_tokens num: {} not equal to audio_feature len: {}",
            n,
            feat_len
        ));
    }
    // contiguous runs of true rows (candle nonzero_slice)
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < flags.len() {
        if flags[i] {
            let s = i;
            while i < flags.len() && flags[i] {
                i += 1;
            }
            runs.push((s, i));
        } else {
            i += 1;
        }
    }
    let d = original.dims()[2];
    let mut out = original;
    let mut sub = 0usize;
    for (s, e) in runs {
        let len = e - s;
        let sub_replace = replace.clone().narrow(0, sub, len).unsqueeze::<3>();
        // all three dims explicit: SliceArg fills missing dims with FULL ranges
        out = out.slice_assign([0..1, s..e, 0..d], sub_replace);
        sub += len;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Weight loading from safetensors (bf16 -> f16), mirroring candle's VarBuilder
// name scheme.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use burn::tensor::TensorData;

pub struct Weights {
    map: HashMap<String, TensorData>,
    device: Device,
}

impl Weights {
    pub fn new(map: HashMap<String, TensorData>, device: Device) -> Self {
        Self { map, device }
    }

    pub fn get<const D: usize>(&self, name: &str) -> Result<Tensor<D>> {
        let td = self
            .map
            .get(name)
            .ok_or_else(|| anyhow!("missing weight: {name}"))?;
        Ok(Tensor::from_data(td.clone(), (&self.device, dt())))
    }
}

pub fn build_model(w: &Weights, cfg: &crate::config::ThinkerConfig) -> Result<Qwen3AsrModel> {
    let device = w.device.clone();
    let ac = &cfg.audio_config;
    let tc = &cfg.text_config;

    // audio tower
    let mut layers = Vec::new();
    for i in 0..ac.encoder_layers {
        let pp = format!("thinker.audio_tower.layers.{i}");
        layers.push(AudioEncoderLayer {
            q_w: w.get(&format!("{pp}.self_attn.q_proj.weight"))?,
            q_b: w.get(&format!("{pp}.self_attn.q_proj.bias"))?,
            k_w: w.get(&format!("{pp}.self_attn.k_proj.weight"))?,
            k_b: w.get(&format!("{pp}.self_attn.k_proj.bias"))?,
            v_w: w.get(&format!("{pp}.self_attn.v_proj.weight"))?,
            v_b: w.get(&format!("{pp}.self_attn.v_proj.bias"))?,
            o_w: w.get(&format!("{pp}.self_attn.out_proj.weight"))?,
            o_b: w.get(&format!("{pp}.self_attn.out_proj.bias"))?,
            attn_ln_w: w.get(&format!("{pp}.self_attn_layer_norm.weight"))?,
            attn_ln_b: w.get(&format!("{pp}.self_attn_layer_norm.bias"))?,
            fc1_w: w.get(&format!("{pp}.fc1.weight"))?,
            fc1_b: w.get(&format!("{pp}.fc1.bias"))?,
            fc2_w: w.get(&format!("{pp}.fc2.weight"))?,
            fc2_b: w.get(&format!("{pp}.fc2.bias"))?,
            final_ln_w: w.get(&format!("{pp}.final_layer_norm.weight"))?,
            final_ln_b: w.get(&format!("{pp}.final_layer_norm.bias"))?,
            d_model: ac.d_model,
            heads: ac.encoder_attention_heads,
            head_dim: ac.d_model / ac.encoder_attention_heads,
        });
    }
    let head_dim = tc.head_dim;
    let eps = tc.rms_norm_eps;
    let audio = AudioEncoder {
        device: device.clone(),
        n_window: ac.n_window,
        conv_chunksize: ac.conv_chunksize,
        conv2d1_w: w.get("thinker.audio_tower.conv2d1.weight")?,
        conv2d1_b: w.get("thinker.audio_tower.conv2d1.bias")?,
        conv2d2_w: w.get("thinker.audio_tower.conv2d2.weight")?,
        conv2d2_b: w.get("thinker.audio_tower.conv2d2.bias")?,
        conv2d3_w: w.get("thinker.audio_tower.conv2d3.weight")?,
        conv2d3_b: w.get("thinker.audio_tower.conv2d3.bias")?,
        conv_out_w: w.get("thinker.audio_tower.conv_out.weight")?,
        proj1_w: w.get("thinker.audio_tower.proj1.weight")?,
        proj1_b: w.get("thinker.audio_tower.proj1.bias")?,
        proj2_w: w.get("thinker.audio_tower.proj2.weight")?,
        proj2_b: w.get("thinker.audio_tower.proj2.bias")?,
        ln_post_w: w.get("thinker.audio_tower.ln_post.weight")?,
        ln_post_b: w.get("thinker.audio_tower.ln_post.bias")?,
        layers,
        inv_freq: compute_default_rope_parameters(ac.d_model, 10000.0),
    };

    // text decoder
    let mut layers = Vec::new();
    for i in 0..tc.num_hidden_layers {
        let pp = format!("thinker.model.layers.{i}");
        layers.push(DecoderLayer {
            q_w: w.get(&format!("{pp}.self_attn.q_proj.weight"))?,
            k_w: w.get(&format!("{pp}.self_attn.k_proj.weight"))?,
            v_w: w.get(&format!("{pp}.self_attn.v_proj.weight"))?,
            o_w: w.get(&format!("{pp}.self_attn.o_proj.weight"))?,
            q_norm_w: w.get(&format!("{pp}.self_attn.q_norm.weight"))?,
            k_norm_w: w.get(&format!("{pp}.self_attn.k_norm.weight"))?,
            gate_w: w.get(&format!("{pp}.mlp.gate_proj.weight"))?,
            up_w: w.get(&format!("{pp}.mlp.up_proj.weight"))?,
            down_w: w.get(&format!("{pp}.mlp.down_proj.weight"))?,
            in_ln_w: w.get(&format!("{pp}.input_layernorm.weight"))?,
            post_ln_w: w.get(&format!("{pp}.post_attention_layernorm.weight"))?,
            n_heads: tc.num_attention_heads,
            n_kv_heads: tc.num_key_value_heads,
            head_dim,
            eps,
            kv_cache: None,
        });
    }
    let text = TextModel {
        embed_tokens: w.get("thinker.model.embed_tokens.weight")?,
        layers,
        norm_w: w.get("thinker.model.norm.weight")?,
        inv_freq: compute_default_rope_parameters(head_dim, tc.rope_theta),
        mrope_section: tc.rope_scaling.mrope_section.clone(),
        eps,
        device: device.clone(),
    };

    Ok(Qwen3AsrModel {
        audio,
        text,
        audio_token_id: cfg.audio_token_id,
        device,
    })
}

/// inv_freq = 1 / base^(2i/dim) for i in 0..dim/2 (candle compute_default_rope_parameters).
fn compute_default_rope_parameters(dim: usize, base: f32) -> Vec<f32> {
    (0..dim)
        .step_by(2)
        .map(|i| 1.0_f32 / base.powf(i as f32 / dim as f32))
        .collect()
}
