//! Quantized MiniCPM5 model implementation (GGUF, for the official `candle` crate).
//!
//! MiniCPM5 is a LLaMA-architecture model with one non-standard twist:
//! `head_dim (128) != hidden_size / num_heads (1536/16 = 96)`, so the Q/K/V
//! projection output (`num_heads * head_dim = 2048`) differs from the hidden
//! size (`1536`). The upstream `candle_transformers::models::quantized_llama`
//! assumes `head_dim == hidden/heads` and `num_heads*head_dim == hidden`, both
//! of which break on MiniCPM5.
//!
//! This module is a vendored copy of `quantized_llama` with these patches:
//! 1. `head_dim` is read from the GGUF (`llama.rope.dimension_count`) / config
//!    instead of `embedding_length / head_count`.
//! 2. The attention output is reshaped to `num_heads * head_dim` (not `hidden`)
//!    before the output projection maps it back to `hidden`.
//! 3. RoPE is always NEOX (non-interleaved / half-split), matching the HF
//!    `LlamaForCausalLM` reference (`rotate_half`). Real MiniCPM5-1B GGUFs
//!    declare `general.architecture = "llama"`, which would select NORM
//!    (interleaved) RoPE from the arch string and silently corrupt attention.
//!
//! Also adds a bf16 safetensors loader (`from_safetensors_dir` → `from_vb`)
//! that quantizes HF-layout weights in memory — `QTensor::quantize_onto`
//! requires a CPU-mapped VarBuilder source, so the safetensors are mmapped on
//! CPU first.
//!
//! Loads weights from a bf16 safetensors directory via
//! `ModelWeights::from_safetensors_dir` (in-memory quantization to a
//! `GgmlDType` such as Q8_0/Q4_K at load time; GGUF files are not supported).
//!

use std::collections::HashMap;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_transformers::quantized_nn::RmsNorm;
use serde::Deserialize;

pub const MAX_SEQ_LEN: usize = 4096;

// QMatMul wrapper adding some tracing.
#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                // In order to extract topk, we extract the data from the tensor and manipulate it
                // directly. Maybe we will want to use some custom ops instead at some point.
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                // top_x contains the row indexes to evaluate for each expert.
                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    // Index the correct hidden states and compute the expert hidden state for
                    // the current expert. We need to make sure to multiply the output hidden
                    // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// RoPE convention: true = NEOX (non-interleaved, pairs i with i+d/2),
    /// false = NORM (interleaved, pairs 2i with 2i+1).
    /// Must match the model architecture — using the wrong convention corrupts
    /// attention patterns and causes severe output degradation.
    rope_is_neox: bool,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let x = x.contiguous()?;
        if self.rope_is_neox {
            candle_nn::rotary_emb::rope(&x, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope_i(&x, &cos, &sin)
        }
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            // This call to contiguous ensures that the fast kernel can be called below. It's
            // actually a no-op except when processing the initial prompt so has no significant
            // impact on performance.
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let y = if q.device().is_metal() && seq_len == 1 {
            // SDPA will do MQA for us
            candle_nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                false,
                1. / (self.head_dim as f32).sqrt(),
                1.,
            )?
        } else {
            // Support for MQA, useful for 70B models and mistral.
            let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?
        };

        let y = y
            .transpose(1, 2)?
            .reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

/// MiniCPM5 config (subset read from config.json).
#[derive(Debug, Clone, Deserialize)]
pub struct MiniCPM5Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    /// Mask cache keyed by (seq_len, kv_len).
    /// kv_len = index_pos + seq_len, so the mask is rectangular when prefix
    /// KV cache entries exist (index_pos > 0).
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    /// Load + quantize from a directory of bf16 safetensors (HF layout) +
    /// config.json. Avoids a pre-converted GGUF — weights are quantized to
    /// `dtype` (e.g. Q8_0) in memory at load time. `device` is the inference
    /// device (Metal); quantization itself runs on CPU (candle's requirement).
    pub fn from_safetensors_dir(dir: &str, dtype: GgmlDType, device: &Device) -> Result<Self> {
        let cfg_bytes = std::fs::read(format!("{dir}/config.json"))
            .map_err(|e| candle_core::Error::msg(format!("config.json: {e}")))?;
        let cfg: MiniCPM5Config = serde_json::from_slice(&cfg_bytes)
            .map_err(|e| candle_core::Error::msg(format!("config.json parse: {e}")))?;
        let mut st_files: Vec<std::path::PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir)
            .map_err(|e| candle_core::Error::msg(format!("readdir {dir}: {e}")))?
        {
            let p = entry
                .map_err(|e| candle_core::Error::msg(format!("dirent: {e}")))?
                .path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                st_files.push(p);
            }
        }
        // Quantize_onto needs a CPU source tensor, so mmap on CPU.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&st_files, DType::BF16, &Device::Cpu)
                .map_err(|e| candle_core::Error::msg(format!("mmap safetensors: {e}")))?
        };
        Self::from_vb(&cfg, &vb, dtype, device)
    }

    fn from_vb(
        cfg: &MiniCPM5Config,
        vb: &VarBuilder,
        dtype: GgmlDType,
        device: &Device,
    ) -> Result<Self> {
        let head_count = cfg.num_attention_heads;
        let head_count_kv = cfg.num_key_value_heads;
        let block_count = cfg.num_hidden_layers;
        let emb = cfg.hidden_size;
        let rope_dim = cfg.head_dim;
        let q_dim = head_count * rope_dim;
        let kv_dim = head_count_kv * rope_dim;
        let inter = cfg.intermediate_size;
        let rms_norm_eps = cfg.rms_norm_eps;
        let rope_freq_base = cfg.rope_theta;
        // MiniCPM5 uses NEOX-style (non-interleaved) RoPE — confirmed against the
        // reference implementation (aha `apply_rotary_pos_emb` → `rotate_half`,
        // the NEOX half-split rotation). Using interleaved (NORM) RoPE here
        // corrupts the position encoding and causes the repetition / missing-EOS
        // failure mode regardless of quantization.
        let rope_is_neox = true;

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Quantize a 2-D weight from CPU (vb) onto `device`.
        let q2 = |name: &str, s: (usize, usize)| -> Result<QTensor> {
            let t = vb.get(s, name)?;
            QTensor::quantize_onto(&t, dtype, device)
        };
        // Quantize a 1-D norm vector.
        let q1 = |name: &str, n: usize| -> Result<QTensor> {
            let t = vb.get((n,), name)?;
            QTensor::quantize_onto(&t, dtype, device)
        };

        let tok_emb_t = vb.get((cfg.vocab_size, emb), "model.embed_tokens.weight")?;
        let tok_emb_q = QTensor::quantize_onto(&tok_emb_t, dtype, device)?;
        let tok_embeddings = tok_emb_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(q1("model.norm.weight", emb)?, rms_norm_eps)?;
        let output_q = match vb.get((cfg.vocab_size, emb), "lm_head.weight") {
            Ok(t) => QTensor::quantize_onto(&t, dtype, device)?,
            Err(_) => tok_emb_q, // tied: reuse the embedding
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let p = format!("model.layers.{layer_idx}");
            let attention_wq = q2(&format!("{p}.self_attn.q_proj.weight"), (q_dim, emb))?;
            let attention_wk = q2(&format!("{p}.self_attn.k_proj.weight"), (kv_dim, emb))?;
            let attention_wv = q2(&format!("{p}.self_attn.v_proj.weight"), (kv_dim, emb))?;
            let attention_wo = q2(&format!("{p}.self_attn.o_proj.weight"), (emb, q_dim))?;
            let feed_forward_w1 = q2(&format!("{p}.mlp.gate_proj.weight"), (inter, emb))?;
            let feed_forward_w2 = q2(&format!("{p}.mlp.down_proj.weight"), (emb, inter))?;
            let feed_forward_w3 = q2(&format!("{p}.mlp.up_proj.weight"), (inter, emb))?;
            let attention_norm = q1(&format!("{p}.input_layernorm.weight"), emb)?;
            let ffn_norm = q1(&format!("{p}.post_attention_layernorm.weight"), emb)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe: MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                }),
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: rope_dim,
                rope_is_neox,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            });
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, emb),
            layers,
            norm,
            output: QMatMul::from_qtensor(output_q)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    /// Build a causal attention mask of shape `(seq_len, kv_len)` where
    /// `kv_len = index_pos + seq_len`.
    ///
    /// When `index_pos == 0` the mask is square `(seq_len, seq_len)` — the
    /// classic case with an empty KV cache.
    ///
    /// When `index_pos > 0` the KV cache already holds `index_pos` entries from
    /// a previously fed prefix.  The mask becomes rectangular: the first
    /// `index_pos` columns are all 0 (every query attends to every prefix key)
    /// and the remaining `seq_len` columns form the standard causal triangle
    /// (query at global position `index_pos + i` cannot attend to keys at global
    /// positions `> index_pos + i`).
    ///
    /// # Shape example  (index_pos=65, seq_len=4)
    /// ```text
    ///              kv 0..64 (prefix)   kv 65  kv 66  kv 67  kv 68
    /// query 65:       0  0 … 0           0      1      1      1
    /// query 66:       0  0 … 0           0      0      1      1
    /// query 67:       0  0 … 0           0      0      0      1
    /// query 68:       0  0 … 0           0      0      0      0
    /// ```
    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    /// Clear the KV cache across all layers.
    ///
    /// Call this between independent conversations to free cached attention
    /// state without recreating the model.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result};
    use candle_transformers::utils::build_causal_mask;

    // ── Mask shape tests ──────────────────────────────────────────────────────

    /// Classic square mask: index_pos=0 produces (seq_len, seq_len).
    #[test]
    fn causal_mask_square_shape() -> Result<()> {
        let mask = build_causal_mask(4, 0, &Device::Cpu)?;
        assert_eq!(mask.dims(), [4, 4]);
        Ok(())
    }

    /// Rectangular mask: index_pos=N produces (seq_len, N + seq_len).
    #[test]
    fn causal_mask_rectangular_shape() -> Result<()> {
        let mask = build_causal_mask(4, 65, &Device::Cpu)?;
        assert_eq!(mask.dims(), [4, 69]);
        Ok(())
    }

    // ── Mask value tests ──────────────────────────────────────────────────────

    /// Square mask values: standard lower-triangular pattern (0=attend, 1=block).
    ///
    /// For seq_len=3, index_pos=0:
    ///   row 0 (global pos 0): attend to pos 0             → [0, 1, 1]
    ///   row 1 (global pos 1): attend to pos 0..1           → [0, 0, 1]
    ///   row 2 (global pos 2): attend to pos 0..2           → [0, 0, 0]
    #[test]
    fn causal_mask_square_values() -> Result<()> {
        let mask = build_causal_mask(3, 0, &Device::Cpu)?;
        let data: Vec<u8> = mask.flatten_all()?.to_vec1()?;
        assert_eq!(data, [0, 1, 1, 0, 0, 1, 0, 0, 0]);
        Ok(())
    }

    /// Rectangular mask values: prefix columns are all-zero, user columns
    /// form the causal triangle.
    ///
    /// For seq_len=3, index_pos=2 → kv_len=5:
    ///   row 0 (global pos 2): attend to kv 0..2  → [0,0, 0,1,1]
    ///   row 1 (global pos 3): attend to kv 0..3  → [0,0, 0,0,1]
    ///   row 2 (global pos 4): attend to kv 0..4  → [0,0, 0,0,0]
    #[test]
    fn causal_mask_rectangular_values() -> Result<()> {
        let mask = build_causal_mask(3, 2, &Device::Cpu)?;
        let data: Vec<u8> = mask.flatten_all()?.to_vec1()?;
        #[rustfmt::skip]
        assert_eq!(data, [
            0, 0,  0, 1, 1,
            0, 0,  0, 0, 1,
            0, 0,  0, 0, 0,
        ]);
        Ok(())
    }

    /// A single-token query (seq_len=1) with prefix produces a single row
    /// of all zeros — it can attend to every key including itself.
    #[test]
    fn causal_mask_single_query_with_prefix() -> Result<()> {
        let mask = build_causal_mask(1, 10, &Device::Cpu)?;
        assert_eq!(mask.dims(), [1, 11]);
        let data: Vec<u8> = mask.flatten_all()?.to_vec1()?;
        assert!(
            data.iter().all(|&v| v == 0),
            "single-query mask should be all-zero"
        );
        Ok(())
    }

    // ── Mask broadcast compatibility test ─────────────────────────────────────

    /// Verify the mask can be broadcast to (batch, heads, seq_len, kv_len) —
    /// the exact shape produced by `Q @ K^T` in forward_attn.
    /// This is the broadcast that previously panicked when index_pos > 0.
    #[test]
    fn causal_mask_broadcasts_to_attention_shape() -> Result<()> {
        let batch = 1usize;
        let heads = 8usize;
        let seq_len = 4usize;
        let index_pos = 10usize;

        let mask = build_causal_mask(seq_len, index_pos, &Device::Cpu)?;
        // Simulate the attention score shape Q @ K^T → (batch, heads, seq_len, kv_len)
        let kv_len = index_pos + seq_len;
        let att_shape = &[batch, heads, seq_len, kv_len];
        let broadcasted = mask.broadcast_as(att_shape.as_slice())?;
        assert_eq!(broadcasted.dims(), att_shape);
        Ok(())
    }
}
