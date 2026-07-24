//! Quantized CosyVoice3 LM (GGUF) — QMatMul mirror of the verbatim aha port
//! in `crate::models::qwen2` (which stays untouched). Only the heavy 2-D
//! matmuls (attn q/k/v/o, mlp gate/up/down) run through
//! `candle_core::quantized::QMatMul`; embeddings, norms, biases and the
//! speech head are dequantized and use plain modules.
//!
//! Activations are F32: candle 0.11's Metal quantized-matmul kernels
//! (`kernel_mul_m{v,m}_q4_K_f32` etc.) take an F32 activation buffer — F16
//! activations are NOT supported there (the mm path even asserts F32). This
//! matches the proven `quantized_minicpm5` path, so the norm/rope/attention
//! semantics are identical to the F32 `qwen2` port with no dtype mixing.
//!
//! GGUF layout (CrispASR convert-cosyvoice3-to-gguf.py, llama.cpp-standard
//! names): `token_embd.weight`, `blk.N.{attn_norm,ffn_norm}.weight`,
//! `blk.N.{attn_q,attn_k,attn_v}.{weight,bias}`, `blk.N.attn_output.weight`,
//! `blk.N.{ffn_gate,ffn_up,ffn_down}.weight`, `output_norm.weight`, plus the
//! CosyVoice3-specific `cosyvoice3.speech_embd.weight` and
//! `cosyvoice3.speech_lm_head.weight`. Forward mirrors
//! `qwen2::Qwen2Decoder::forward` exactly (KV cache, GQA eager attention,
//! RoPE theta 1e6 NEOX, RMSNorm eps 1e-6, biased QKV, SwiGLU).

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use candle_core::quantized::{QMatMul, QTensor, gguf_file};
use candle_core::{Device, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, ops};

use crate::common::modules::eager_attention_forward;
use crate::models::qwen2::Qwen2Config;
use crate::position_embed::rope::{RoPE, apply_rotary_pos_emb};

/// One decoder layer, semantically identical to `qwen2::Qwen2DecoderLayer`.
struct QuantizedQwen2Layer {
    q_proj: QMatMul,
    q_bias: Tensor,
    k_proj: QMatMul,
    k_bias: Tensor,
    v_proj: QMatMul,
    v_bias: Tensor,
    o_proj: QMatMul,
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QuantizedQwen2Layer {
    /// Mirror of `qwen2::Qwen2DecoderLayer::forward` + `Qwen2Attention::forward`
    /// (biased QKV, KV cache cat on dim 2, GQA eager attention, residual +
    /// SwiGLU MLP). QMatMul takes/returns F32 here (Metal kernel constraint),
    /// so the whole stream stays F32 like the unquantized port.
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
        let q = self.q_proj.forward(&xs)?.broadcast_add(&self.q_bias)?;
        let k = self.k_proj.forward(&xs)?.broadcast_add(&self.k_bias)?;
        let v = self.v_proj.forward(&xs)?.broadcast_add(&self.v_bias)?;
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin, false)?;
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

/// Quantized Qwen2 backbone, drop-in for `qwen2::Qwen2Decoder` (KV-cached
/// forward + clear_kv_cache, same signature).
pub struct QuantizedQwen2Decoder {
    layers: Vec<QuantizedQwen2Layer>,
    norm: RmsNorm,
    rotary_emb: RoPE,
}

impl QuantizedQwen2Decoder {
    pub fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(1)?;
        let (cos, sin) = self
            .rotary_emb
            .forward(seqlen_offset, seq_len, xs.device())?;
        let mut xs = xs.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask)?;
        }
        Ok(xs.apply(&self.norm)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }
}

/// All weights `CosyVoice3LM` needs from the quantized GGUF.
pub struct QuantizedLmWeights {
    pub token_embedding: Embedding,
    pub decoder: QuantizedQwen2Decoder,
    pub speech_embedding: Embedding,
    pub speech_lm_head: Linear,
    pub speech_vocab: usize,
}

fn load_qtensor(
    ct: &gguf_file::Content,
    reader: &mut File,
    name: &str,
    device: &Device,
) -> Result<QTensor> {
    ct.tensor(reader, name, device)
        .with_context(|| format!("gguf tensor `{name}`"))
}

/// Dequantized F32 tensor (embeddings / norms / biases / speech head).
fn deq(ct: &gguf_file::Content, reader: &mut File, name: &str, device: &Device) -> Result<Tensor> {
    Ok(load_qtensor(ct, reader, name, device)?.dequantize(device)?)
}

fn qmat(
    ct: &gguf_file::Content,
    reader: &mut File,
    name: &str,
    device: &Device,
) -> Result<QMatMul> {
    Ok(QMatMul::from_qtensor(load_qtensor(
        ct, reader, name, device,
    )?)?)
}

/// Load `cosyvoice3-llm-q4_k.gguf`. `cfg` comes from
/// `CosyVoice-BlankEN/config.json` (same source as the F32 path); the GGUF
/// hparams are only sanity-checked.
pub fn load_gguf(path: &Path, cfg: &Qwen2Config, device: &Device) -> Result<QuantizedLmWeights> {
    let mut reader = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let ct = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("parse gguf header {}", path.display()))?;
    if let Some(n_layers) = ct
        .metadata
        .get("cosyvoice3.llm.n_layers")
        .and_then(|v| v.to_u32().ok())
        && n_layers as usize != cfg.num_hidden_layers
    {
        return Err(anyhow!(
            "{}: gguf n_layers {n_layers} != config num_hidden_layers {}",
            path.display(),
            cfg.num_hidden_layers
        ));
    }

    // Dequantized (F32) tensors: embeddings / norms / biases / speech head.
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("blk.{i}");
        layers.push(QuantizedQwen2Layer {
            q_proj: qmat(&ct, &mut reader, &format!("{p}.attn_q.weight"), device)?,
            q_bias: deq(&ct, &mut reader, &format!("{p}.attn_q.bias"), device)?,
            k_proj: qmat(&ct, &mut reader, &format!("{p}.attn_k.weight"), device)?,
            k_bias: deq(&ct, &mut reader, &format!("{p}.attn_k.bias"), device)?,
            v_proj: qmat(&ct, &mut reader, &format!("{p}.attn_v.weight"), device)?,
            v_bias: deq(&ct, &mut reader, &format!("{p}.attn_v.bias"), device)?,
            o_proj: qmat(&ct, &mut reader, &format!("{p}.attn_output.weight"), device)?,
            gate_proj: qmat(&ct, &mut reader, &format!("{p}.ffn_gate.weight"), device)?,
            up_proj: qmat(&ct, &mut reader, &format!("{p}.ffn_up.weight"), device)?,
            down_proj: qmat(&ct, &mut reader, &format!("{p}.ffn_down.weight"), device)?,
            input_layernorm: RmsNorm::new(
                deq(&ct, &mut reader, &format!("{p}.attn_norm.weight"), device)?,
                cfg.rms_norm_eps,
            ),
            post_attention_layernorm: RmsNorm::new(
                deq(&ct, &mut reader, &format!("{p}.ffn_norm.weight"), device)?,
                cfg.rms_norm_eps,
            ),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim,
            kv_cache: None,
        });
    }
    let norm = RmsNorm::new(
        deq(&ct, &mut reader, "output_norm.weight", device)?,
        cfg.rms_norm_eps,
    );
    let rotary_emb = RoPE::new(head_dim, cfg.rope_theta, device)?;

    let token_w = deq(&ct, &mut reader, "token_embd.weight", device)?;
    let speech_w = deq(&ct, &mut reader, "cosyvoice3.speech_embd.weight", device)?;
    let head_w = deq(&ct, &mut reader, "cosyvoice3.speech_lm_head.weight", device)?;
    let (speech_vocab, speech_dim) = head_w.dims2().context("speech_lm_head.weight dims")?;
    if speech_dim != cfg.hidden_size {
        return Err(anyhow!(
            "speech_lm_head.weight in_dim {speech_dim} != hidden_size {}",
            cfg.hidden_size
        ));
    }
    Ok(QuantizedLmWeights {
        token_embedding: Embedding::new(token_w, cfg.hidden_size),
        decoder: QuantizedQwen2Decoder {
            layers,
            norm,
            rotary_emb,
        },
        speech_embedding: Embedding::new(speech_w, cfg.hidden_size),
        speech_lm_head: Linear::new(head_w, None),
        speech_vocab,
    })
}
