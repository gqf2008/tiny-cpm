//! Ported from aha (github.com/jhqxxx/aha) src/models/fun_asr_nano/model.rs
use anyhow::{Result, anyhow, bail};
use candle_core::{D, Device, IndexOp, Tensor};
use candle_core::quantized::GgmlDType;
use candle_nn::{
    Conv1d, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding, linear,
    linear_no_bias, rms_norm, ops::softmax_last_dim,
};

use crate::{
    common::{
        InferenceModel,
        modules::{
            NaiveAttention, TwoLinearMLP, conv1d_depthwise, eager_attention_forward, get_conv1d,
            get_layer_norm,
        },
    },
    models::{
        fun_asr_nano::config::FunASRNanoConfig,
        qwen3::{config::Qwen3Config, model::Qwen3Model},
        qwen3_tts::quantized_talker,
    },
    position_embed::{rope::RoPE, sinusoidal::SinusoidalPositionEncoderCat},
    utils::tensor_utils::{
        attn_masked_fill, get_equal_mask, masked_scatter_dim0, prepare_causal_attention_mask,
    },
};

pub struct MultiHeadedAttentionSANM {
    head_dim: usize,
    n_head: usize,
    linear_out: Linear,
    linear_q_k_v: Linear,
    fsmn_block: Conv1d,
    left_padding: usize,
    right_padding: usize,
    scaling: f64,
}

impl MultiHeadedAttentionSANM {
    pub fn new(
        vb: VarBuilder,
        n_head: usize,
        in_dim: usize,
        hidden_dim: usize,
        kernel_size: usize,
        sanm_shfit: usize,
    ) -> Result<Self> {
        let head_dim = hidden_dim / n_head;
        let linear_out = linear(hidden_dim, hidden_dim, vb.pp("linear_out"))?;
        let linear_q_k_v = linear(in_dim, hidden_dim * 3, vb.pp("linear_q_k_v"))?;
        let fsmn_block = get_conv1d(
            vb.pp("fsmn_block"),
            hidden_dim,
            hidden_dim,
            kernel_size,
            0,
            1,
            1,
            hidden_dim,
            false,
        )?;
        let mut left_padding = (kernel_size - 1) / 2;
        if sanm_shfit > 0 {
            left_padding += sanm_shfit;
        }
        let right_padding = kernel_size - 1 - left_padding;
        let scaling = (head_dim as f64).powf(-0.5);
        Ok(Self {
            head_dim,
            n_head,
            linear_out,
            linear_q_k_v,
            fsmn_block,
            left_padding,
            right_padding,
            scaling,
        })
    }

    pub fn forward_fsmn(
        &self,
        inputs: &Tensor,
        mask: Option<&Tensor>,
        mask_shfit_chunk: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut inputs = inputs.clone();
        let mask = if let Some(mask) = mask {
            let mut mask = mask.unsqueeze(D::Minus1)?.unsqueeze(0)?;
            if let Some(mask_shfit_chunk) = mask_shfit_chunk {
                mask = mask.broadcast_mul(mask_shfit_chunk)?;
            }
            inputs = inputs.broadcast_mul(&mask)?;
            Some(mask)
        } else {
            None
        };
        let xs = inputs.transpose(1, 2)?;
        let xs = xs.pad_with_zeros(D::Minus1, self.left_padding, self.right_padding)?;
        // let xs = self.fsmn_block.forward(&xs)?;
        let xs = conv1d_depthwise(&xs, self.fsmn_block.weight(), self.fsmn_block.bias())?;
        let xs = xs.transpose(1, 2)?;
        let mut xs = xs.add(&inputs)?;
        if let Some(mask) = mask {
            xs = xs.broadcast_mul(&mask)?;
        }
        Ok(xs)
    }
    pub fn forward_qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, t, _) = xs.dims3()?;
        let q_k_v = self
            .linear_q_k_v
            .forward(xs)?
            .reshape((b, t, 3, self.n_head, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let q_h = q_k_v.i(0)?.contiguous()?;
        let k_h = q_k_v.i(1)?.contiguous()?;
        let v_h = q_k_v.i(2)?.contiguous()?;
        let v = v_h.transpose(1, 2)?.reshape((b, t, ()))?;
        Ok((q_h, k_h, v_h, v))
    }

    pub fn forward_attention(
        &self,
        values: &Tensor,
        scores: &Tensor,
        mask: Option<&Tensor>,
        mask_att_chunk_encoder: Option<&Tensor>,
    ) -> Result<Tensor> {
        let bs = scores.dim(0)?;
        let attn = if let Some(mask) = mask {
            let mask = if let Some(mask_att_chunk_encoder) = mask_att_chunk_encoder {
                mask.mul(mask_att_chunk_encoder)?
            } else {
                mask.clone()
            };
            // mask: rank = 2
            let mask = get_equal_mask(&mask, 0)?;
            let scores = attn_masked_fill(scores, &mask, f32::NEG_INFINITY)?;
            let attn = softmax_last_dim(&scores)?;
            attn_masked_fill(&attn, &mask, 0.0)?
        } else {
            softmax_last_dim(scores)?
        };
        let xs = attn.matmul(values)?;
        let xs =
            xs.transpose(1, 2)?
                .contiguous()?
                .reshape((bs, (), self.n_head * self.head_dim))?;
        let xs = self.linear_out.forward(&xs)?;
        Ok(xs)
    }

    pub fn forward_simple(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let q_k_v = self.linear_q_k_v.forward(xs)?;
        let dim = self.head_dim * self.n_head;
        let q_h = q_k_v
            .narrow(D::Minus1, 0, dim)?
            .reshape((b, t, self.n_head, ()))?
            .permute((0, 2, 1, 3))?;
        let k_h = q_k_v
            .narrow(D::Minus1, dim, dim)?
            .reshape((b, t, self.n_head, ()))?
            .permute((0, 2, 1, 3))?;
        let v = q_k_v.narrow(D::Minus1, dim * 2, dim)?;
        let v_h = v.reshape((b, t, self.n_head, ()))?.permute((0, 2, 1, 3))?;
        let fsmn_memory = v.transpose(1, 2)?;
        let fsmn_memory = fsmn_memory
            .pad_with_zeros(D::Minus1, self.left_padding, self.right_padding)?
            .contiguous()?;
        let fsmn_memory = self.fsmn_block.forward(&fsmn_memory)?;
        // let fsmn_memory = conv1d_group_parallel(&fsmn_memory, &self.fsmn_block)?;

        let fsmn_memory = fsmn_memory.transpose(1, 2)?;
        let fsmn_memory = fsmn_memory.add(&v)?;
        let att_outs = eager_attention_forward(&q_h, &k_h, &v_h, None, None, self.scaling)?;
        let att_outs = att_outs.reshape((b, t, ()))?;
        let att_outs = self.linear_out.forward(&att_outs)?;
        let att_outs = att_outs.add(&fsmn_memory)?;
        Ok(att_outs)
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        mask_shfit_chunk: Option<&Tensor>,
        mask_att_chunk_encoder: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (q_h, k_h, v_h, v) = self.forward_qkv(xs)?;
        let fsmn_memory = self.forward_fsmn(&v, mask, mask_shfit_chunk)?;
        let q_h = q_h.affine(self.scaling, 0.0)?;
        let scores = q_h.matmul(&k_h.transpose(D::Minus2, D::Minus1)?)?;
        let attn_outs = self.forward_attention(&v_h, &scores, mask, mask_att_chunk_encoder)?;
        let att_outs = attn_outs.add(&fsmn_memory)?;
        Ok(att_outs)
    }
}

pub struct EncoderLayerSANM {
    self_attn: MultiHeadedAttentionSANM,
    feed_forward: TwoLinearMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
    concat_linear: Option<Linear>,
    normalize_before: bool,
    in_dim: usize,
    hidden_dim: usize,
}

impl EncoderLayerSANM {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        hidden_dim: usize,
        n_head: usize,
        kernel_size: usize,
        sanm_shfit: usize,
        hidden_units: usize,
        normalize_before: bool,
        concat_after: bool,
    ) -> Result<Self> {
        let self_attn = MultiHeadedAttentionSANM::new(
            vb.pp("self_attn"),
            n_head,
            in_dim,
            hidden_dim,
            kernel_size,
            sanm_shfit,
        )?;
        let feed_forward = TwoLinearMLP::new(
            vb.pp("feed_forward"),
            hidden_dim,
            hidden_units,
            hidden_dim,
            candle_nn::Activation::Relu,
            true,
            "w_1",
            "w_2",
        )?;
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-5, in_dim, true)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-5, hidden_dim, true)?;
        let concat_linear = if concat_after {
            let lin = linear(hidden_dim * 2, hidden_dim, vb.pp("concat_linear"))?;
            Some(lin)
        } else {
            None
        };
        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            concat_linear,
            normalize_before,
            in_dim,
            hidden_dim,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        mask_shfit_chunk: Option<&Tensor>,
        mask_att_chunk_encoder: Option<&Tensor>,
    ) -> Result<Tensor> {
        let stoch_layer_coeff = 1.0f64;
        let residual = xs.clone();
        let mut xs = if self.normalize_before {
            self.norm1.forward(xs)?
        } else {
            xs.clone()
        };
        if self.concat_linear.is_some() {
            let attn =
                self.self_attn
                    .forward(&xs, mask, mask_shfit_chunk, mask_att_chunk_encoder)?;
            let x_concat = Tensor::cat(&[&xs, &attn], D::Minus1)?;
            if self.in_dim == self.hidden_dim
                && let Some(concat_linear) = &self.concat_linear
            {
                let x_concat = concat_linear
                    .forward(&x_concat)?
                    .affine(stoch_layer_coeff, 0.0)?;
                xs = residual.add(&x_concat)?;
            } else if let Some(concat_linear) = &self.concat_linear {
                xs = concat_linear
                    .forward(&x_concat)?
                    .affine(stoch_layer_coeff, 0.0)?;
            }
        } else if self.in_dim == self.hidden_dim {
            let attn = self
                .self_attn
                .forward(&xs, mask, mask_shfit_chunk, mask_att_chunk_encoder)?
                .affine(stoch_layer_coeff, 0.0)?;
            xs = residual.add(&attn)?;
        } else {
            xs = self
                .self_attn
                .forward(&xs, mask, mask_shfit_chunk, mask_att_chunk_encoder)?
                .affine(stoch_layer_coeff, 0.0)?;
        }

        if !self.normalize_before {
            xs = self.norm1.forward(&xs)?;
        }
        let residual = xs.clone();
        if self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        xs = self
            .feed_forward
            .forward(&xs)?
            .affine(stoch_layer_coeff, 0.0)?;
        xs = residual.add(&xs)?;
        if !self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        Ok(xs)
    }

    pub fn forward_simple(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.norm1.forward(xs)?;
        if self.in_dim == self.hidden_dim {
            let attn = self.self_attn.forward_simple(&xs)?;
            xs = residual.add(&attn)?;
        } else {
            xs = self.self_attn.forward_simple(&xs)?;
        }

        let residual = xs.clone();
        let xs = self.norm2.forward(&xs)?;

        let xs = self.feed_forward.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct SenseVoiceEncoderSmall {
    embed: SinusoidalPositionEncoderCat,
    encoders0: EncoderLayerSANM,
    encoders: Vec<EncoderLayerSANM>,
    tp_encoders: Vec<EncoderLayerSANM>,
    after_norm: LayerNorm,
    tp_norm: LayerNorm,
    scaling: f64,
}

impl SenseVoiceEncoderSmall {
    pub fn new(
        vb: VarBuilder,
        input_size: usize,
        output_size: usize,
        attention_heads: usize,
        linear_units: usize,
        num_blocks: usize,
        tp_blocks: usize,
        normalize_before: bool,
        kernel_size: usize,
        sanm_shfit: usize,
    ) -> Result<Self> {
        let embed = SinusoidalPositionEncoderCat::new(Some(input_size), true, vb.device())?;

        let encoders0 = EncoderLayerSANM::new(
            vb.pp("encoders0.0"),
            input_size,
            output_size,
            attention_heads,
            kernel_size,
            sanm_shfit,
            linear_units,
            normalize_before,
            false,
        )?;
        let mut encoders = vec![];
        let vb_encoders = vb.pp("encoders");
        for i in 0..(num_blocks - 1) {
            let encoder_i = EncoderLayerSANM::new(
                vb_encoders.pp(i),
                output_size,
                output_size,
                attention_heads,
                kernel_size,
                sanm_shfit,
                linear_units,
                normalize_before,
                false,
            )?;
            encoders.push(encoder_i);
        }
        let vb_tp_encoders = vb.pp("tp_encoders");
        let mut tp_encoders = vec![];
        for i in 0..tp_blocks {
            let tp_blocks_i = EncoderLayerSANM::new(
                vb_tp_encoders.pp(i),
                output_size,
                output_size,
                attention_heads,
                kernel_size,
                sanm_shfit,
                linear_units,
                normalize_before,
                false,
            )?;
            tp_encoders.push(tp_blocks_i);
        }
        let after_norm = get_layer_norm(vb.pp("after_norm"), 1e-5, output_size, true)?;
        let tp_norm = get_layer_norm(vb.pp("tp_norm"), 1e-5, output_size, true)?;
        let scaling = (output_size as f64).powf(0.5);
        Ok(Self {
            embed,
            encoders0,
            encoders,
            tp_encoders,
            after_norm,
            tp_norm,
            scaling,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.affine(self.scaling, 0.0)?;
        let xs = self.embed.forward(&xs, 0)?;
        let mut xs = self.encoders0.forward_simple(&xs)?;
        for encoder_layer in &self.encoders {
            xs = encoder_layer.forward_simple(&xs)?;
        }
        xs = self.after_norm.forward(&xs)?;
        for tp_layer in &self.tp_encoders {
            xs = tp_layer.forward_simple(&xs)?;
        }
        xs = self.tp_norm.forward(&xs)?;
        Ok(xs)
    }
}

pub struct AdaptorEncoderLayer {
    self_attn: NaiveAttention,
    feed_forward: TwoLinearMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
    concat_linear: Option<Linear>,
    normalize_before: bool,
}

impl AdaptorEncoderLayer {
    pub fn new(
        vb: VarBuilder,
        llm_dim: usize,
        n_head: usize,
        normalize_before: bool,
        concat_after: bool,
    ) -> Result<Self> {
        let self_attn = NaiveAttention::new(
            vb.pp("self_attn"),
            llm_dim,
            n_head,
            n_head,
            None,
            true,
            Some("linear_q"),
            Some("linear_k"),
            Some("linear_v"),
            Some("linear_out"),
        )?;
        let feed_forward = TwoLinearMLP::new(
            vb.pp("feed_forward"),
            llm_dim,
            llm_dim / 4,
            llm_dim,
            candle_nn::Activation::Relu,
            true,
            "w_1",
            "w_2",
        )?;
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-5, llm_dim, true)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-5, llm_dim, true)?;
        let concat_linear = if concat_after {
            let lin = linear(llm_dim * 2, llm_dim, vb.pp("concat_linear"))?;
            Some(lin)
        } else {
            None
        };
        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            concat_linear,
            normalize_before,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let stoch_layer_coeff = 1.0f64;
        let residual = xs.clone();
        let mut xs = if self.normalize_before {
            self.norm1.forward(xs)?
        } else {
            xs.clone()
        };
        if let Some(concat_linear) = &self.concat_linear {
            let attn = self.self_attn.forward(&xs, None, None, mask, false)?;
            let x_concat = Tensor::cat(&[&xs, &attn], D::Minus1)?;
            let x_concat = concat_linear
                .forward(&x_concat)?
                .affine(stoch_layer_coeff, 0.0)?;
            xs = residual.add(&x_concat)?;
        } else {
            let attn = self
                .self_attn
                .forward(&xs, None, None, mask, false)?
                .affine(stoch_layer_coeff, 0.0)?;
            xs = residual.add(&attn)?;
        }
        if !self.normalize_before {
            xs = self.norm1.forward(&xs)?;
        }
        let residual = xs.clone();
        if self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        xs = self
            .feed_forward
            .forward(&xs)?
            .affine(stoch_layer_coeff, 0.0)?;
        xs = residual.add(&xs)?;
        if !self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        Ok(xs)
    }
}

pub struct AudioAdaptor {
    k: usize,
    linear1: Linear,
    linear2: Linear,
    blocks: Vec<AdaptorEncoderLayer>,
}

impl AudioAdaptor {
    pub fn new(
        vb: VarBuilder,
        downsample_rate: usize,
        encoder_dim: usize,
        llm_dim: usize,
        ffn_dim: usize,
        n_layer: usize,
        attention_heads: usize,
    ) -> Result<Self> {
        let linear1 = linear(encoder_dim * downsample_rate, ffn_dim, vb.pp("linear1"))?;
        let linear2 = linear(ffn_dim, llm_dim, vb.pp("linear2"))?;
        let mut blocks = vec![];
        let vb_blocks = vb.pp("blocks");
        for i in 0..n_layer {
            let layer =
                AdaptorEncoderLayer::new(vb_blocks.pp(i), llm_dim, attention_heads, true, false)?;
            blocks.push(layer);
        }
        Ok(Self {
            k: downsample_rate,
            linear1,
            linear2,
            blocks,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, dim) = xs.dims3()?;
        let chunk_num = (seq_len - 1) / self.k + 1;
        let pad_num = chunk_num * self.k - seq_len;
        let xs = xs.pad_with_zeros(1, 0, pad_num)?;
        let xs = xs.contiguous()?.reshape((bs, chunk_num, dim * self.k))?;
        let xs = self.linear1.forward(&xs)?.relu()?;
        let mut xs = self.linear2.forward(&xs)?;
        for block in &self.blocks {
            xs = block.forward(&xs, None)?;
        }
        Ok(xs)
    }
}

/// Quant-only mirror of the Qwen3 LLM fields Fun-ASR needs, with a
/// `QuantizedTalkerBackbone` in place of `Vec<Qwen3DecoderLayer>`. Lives here
/// (not in `qwen3/model.rs`) because the `qwen3` leaf cannot depend on
/// `qwen3_tts` (cycle: `qwen3_tts -> qwen3`). The forward mirrors
/// `Qwen3Model::forward_hidden` + `forward` exactly; the backbone returns raw
/// (unnormed) F32 output, cast to the embed dtype before the single `norm` +
/// `lm_head` (no double-norm). `kv_pos == seqlen_offset` for the preallocated KV.
pub struct FunAsrQuantLlm {
    embed_tokens: Embedding,
    backbone: quantized_talker::QuantizedTalkerBackbone,
    norm: RmsNorm,
    rotary: RoPE,
    lm_head: Linear,
    kv_cap: usize,
}

impl FunAsrQuantLlm {
    /// `vb_metal` is the `llm`-rooted Metal builder (embed/norm/lm_head stay BF16);
    /// `vb_cpu` is the repo-root CPU builder — `load_stack` prefixes `llm.model`
    /// itself (Fun-ASR's LLM keys are `llm.model.layers.N.*`, per Qwen3Model::new's
    /// auto-`model` prefix).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb_metal: VarBuilder,
        vb_cpu: VarBuilder,
        cfg: &Qwen3Config,
        quant: GgmlDType,
        kv_cap: usize,
        device: &Device,
    ) -> Result<Self> {
        // Mirror Qwen3Model::new's conditional `model.` prefix — Fun-ASR's LLM keys
        // are `llm.model.*`, so embed/norm/lm_head sit one level deeper than the
        // `llm`-rooted vb_metal we receive.
        let vb_metal = if vb_metal.contains_tensor("model.embed_tokens.weight") {
            vb_metal.pp("model")
        } else {
            vb_metal
        };
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_metal.pp("embed_tokens"))?;
        let backbone = quantized_talker::load_stack(
            &vb_cpu,
            "llm.model",
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.intermediate_size,
            cfg.rms_norm_eps,
            quant,
            kv_cap,
            device,
        )?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_metal.pp("norm"))?;
        let rotary = RoPE::new(cfg.head_dim, cfg.rope_theta, device)?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_metal.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            backbone,
            norm,
            rotary,
            lm_head,
            kv_cap,
        })
    }

    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed_tokens.forward(input_ids)?)
    }

    pub fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let inputs_embeds = if let Some(inputs_embeds) = inputs_embeds {
            inputs_embeds.clone()
        } else {
            let input_ids = input_ids.unwrap();
            self.embed_tokens.forward(input_ids)?
        };
        let (bs, seq_len, _) = inputs_embeds.dims3()?;
        let attention_mask: Option<Tensor> = if seq_len <= 1 {
            None
        } else {
            Some(prepare_causal_attention_mask(
                bs,
                seq_len,
                0,
                inputs_embeds.device(),
            )?)
        };
        let (cos, sin) = self
            .rotary
            .forward(seqlen_offset, seq_len, inputs_embeds.device())?;
        if seqlen_offset + seq_len > self.kv_cap {
            bail!(
                "Fun-ASR quant KV cap ({}) exceeded at position {}+{}; raise TINY_CPM_FUNASR_KV_CAP",
                self.kv_cap,
                seqlen_offset,
                seq_len
            );
        }
        let mut hs = self
            .backbone
            .forward(&inputs_embeds, &cos, &sin, attention_mask.as_ref(), seqlen_offset)?;
        // Quant backbone returns F32 (QMatMul constraint); cast back for norm/lm_head.
        hs = hs.to_dtype(inputs_embeds.dtype())?;
        let hs = self.norm.forward(&hs)?;
        let hs = hs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hs)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.backbone.clear_kv_cache();
    }
}

/// Fun-ASR's LLM: either the BF16 `Qwen3Model` (default) or the quantized
/// `FunAsrQuantLlm` (opt-in via `--quant`). Dispatches the three callsites
/// `FunAsrNanoModel` uses (embedding_token_id / forward / clear_kv_cache).
pub enum FunAsrLlm {
    Full(Qwen3Model),
    Quant(FunAsrQuantLlm),
}

impl FunAsrLlm {
    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Full(m) => m.embedding_token_id(input_ids),
            Self::Quant(m) => m.embedding_token_id(input_ids),
        }
    }

    pub fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Full(m) => m.forward(input_ids, inputs_embeds, seqlen_offset),
            Self::Quant(m) => m.forward(input_ids, inputs_embeds, seqlen_offset),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            Self::Full(m) => m.clear_kv_cache(),
            Self::Quant(m) => m.clear_kv_cache(),
        }
    }
}

/// Build the SenseVoice audio encoder + adaptor (shared by the BF16 and quantized
/// constructors — only the LLM stack differs between them).
fn build_audio(
    vb: &VarBuilder,
    config: &FunASRNanoConfig,
) -> Result<(SenseVoiceEncoderSmall, AudioAdaptor)> {
    let input_size = config.frontend_conf.lfr_m * config.frontend_conf.n_mels;
    let audio_encoder = SenseVoiceEncoderSmall::new(
        vb.pp("audio_encoder"),
        input_size,
        config.audio_encoder_conf.output_size,
        config.audio_encoder_conf.attention_heads,
        config.audio_encoder_conf.linear_units,
        config.audio_encoder_conf.num_blocks,
        config.audio_encoder_conf.tp_blocks,
        config.audio_encoder_conf.normalize_before,
        config.audio_encoder_conf.kernel_size,
        config.audio_encoder_conf.sanm_shfit,
    )?;
    let audio_adaptor = AudioAdaptor::new(
        vb.pp("audio_adaptor"),
        config.audio_adaptor_conf.downsample_rate,
        config.audio_adaptor_conf.encoder_dim,
        config.audio_adaptor_conf.llm_dim,
        config.audio_adaptor_conf.ffn_dim,
        config.audio_adaptor_conf.n_layer,
        8,
    )?;
    Ok((audio_encoder, audio_adaptor))
}

pub struct FunAsrNanoModel {
    audio_encoder: SenseVoiceEncoderSmall,
    audio_adaptor: AudioAdaptor,
    llm: FunAsrLlm,
    stop_token_ids: Vec<u32>,
}
impl FunAsrNanoModel {
    pub fn new(
        vb: VarBuilder,
        config: &FunASRNanoConfig,
        llm_cfg: &Qwen3Config,
        eos_ids: Vec<u32>,
    ) -> Result<Self> {
        let (audio_encoder, audio_adaptor) = build_audio(&vb, config)?;
        let llm = FunAsrLlm::Full(Qwen3Model::new(llm_cfg, vb.pp("llm"), vec![])?);
        Ok(Self {
            audio_encoder,
            audio_adaptor,
            llm,
            stop_token_ids: eos_ids,
        })
    }

    /// Quantized LLM stack (opt-in). The audio encoder + adaptor load BF16 from
    /// `vb`; only the LLM decoder layers quantize from the CPU-mmaped `vb_cpu`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_quantized(
        vb: VarBuilder,
        vb_cpu: VarBuilder,
        config: &FunASRNanoConfig,
        llm_cfg: &Qwen3Config,
        eos_ids: Vec<u32>,
        quant: GgmlDType,
        kv_cap: usize,
        device: &Device,
    ) -> Result<Self> {
        let (audio_encoder, audio_adaptor) = build_audio(&vb, config)?;
        let llm = FunAsrLlm::Quant(FunAsrQuantLlm::new(
            vb.pp("llm"),
            vb_cpu,
            llm_cfg,
            quant,
            kv_cap,
            device,
        )?);
        Ok(Self {
            audio_encoder,
            audio_adaptor,
            llm,
            stop_token_ids: eos_ids,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        speech: Option<&Tensor>,
        fbank_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.llm.embedding_token_id(input_ids)?;
        if let Some(speech) = speech
            && let Some(fbank_mask) = fbank_mask
        {
            let speech = self.audio_encoder.forward(speech)?;
            let encoder_out = self.audio_adaptor.forward(&speech)?;
            let speech_token_len = fbank_mask.sum_all()?.to_scalar::<u32>()?;
            let audio_embed = encoder_out
                .squeeze(0)?
                .narrow(0, 0, speech_token_len as usize)?;
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &audio_embed, fbank_mask)?;
        }
        let logits = self
            .llm
            .forward(None, Some(&inputs_embeds), seqlen_offset)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.llm.clear_kv_cache();
    }
}

impl InferenceModel for FunAsrNanoModel {
    fn forward_initial(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        data: crate::common::MultiModalData,
    ) -> Result<Tensor> {
        if data.data_vec.len() != 2 {
            return Err(anyhow!(
                "FunAsrNano process data error, must have speech, fbank_mask"
            ));
        }
        let speech = &data.data_vec[0];
        let fbank_mask = &data.data_vec[1];
        self.forward(
            input_ids,
            speech.as_ref(),
            fbank_mask.as_ref(),
            seqlen_offset,
        )
    }

    fn forward_step(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward(input_ids, None, None, seqlen_offset)
    }

    fn clear_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        self.stop_token_ids.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Minimal Fun-ASR-Nano config.yaml covering every field of FunASRNanoConfig.
    const TEST_CONFIG_YAML: &str = r#"
audio_encoder_conf:
  output_size: 512
  attention_heads: 4
  linear_units: 2048
  num_blocks: 50
  tp_blocks: 20
  dropout_rate: 0.1
  positional_dropout_rate: 0.1
  attention_dropout_rate: 0.0
  input_layer: pe_linear
  pos_enc_class: SinusoidalPositionEncoder
  normalize_before: true
  kernel_size: 11
  sanm_shfit: 0
  selfattention_layer_type: sanm
  freeze: false
  freeze_layer_num: -1
  feat_permute: true
llm_conf:
  hub: hf
  freeze: true
  llm_dtype: bfloat16
  init_param_path: Qwen3-0.6B
audio_adaptor_conf:
  downsample_rate: 2
  use_low_frame_rate: false
  ffn_dim: 2048
  llm_dim: 1024
  encoder_dim: 512
  n_layer: 2
  freeze: false
detach_ctc_decoder: true
ctc_decoder_conf:
  downsample_rate: 2
  ffn_dim: 2048
  llm_dim: 1024
  encoder_dim: 512
  n_layer: 2
  freeze: false
ctc_weight: 0.3
ctc_conf:
  dropout_rate: 0.0
  ctc_type: builtin
  reduce: true
  ignore_nan_grad: true
frontend_conf:
  fs: 16000
  window: hamming
  n_mels: 80
  frame_length: 25
  frame_shift: 10
  lfr_m: 7
  lfr_n: 6
"#;

    #[test]
    fn parse_config_yaml() {
        let cfg: FunASRNanoConfig = serde_yaml::from_str(TEST_CONFIG_YAML).unwrap();
        assert_eq!(cfg.frontend_conf.fs, 16000);
        assert_eq!(cfg.frontend_conf.n_mels, 80);
        assert_eq!(cfg.frontend_conf.lfr_m, 7);
        assert_eq!(cfg.frontend_conf.lfr_n, 6);
        assert_eq!(cfg.llm_conf.llm_dtype, "bfloat16");
        assert_eq!(cfg.audio_encoder_conf.output_size, 512);
        assert_eq!(cfg.audio_adaptor_conf.downsample_rate, 2);
        assert!(cfg.frontend_conf.cmvn_file.is_none());
    }

    /// The fake-token count (fbank_mask sum) must never exceed the adaptor's
    /// chunk count, or the `narrow` in `FunAsrNanoModel::forward` would panic.
    #[test]
    fn fake_token_len_within_adaptor_output() {
        // same arithmetic as FunAsrNanoProcessor::process_info
        // (two stride-2 length reductions, kernel 3 / pad 1)
        let fake_token_len = |speech_lengths: usize| {
            let olens = 1 + (speech_lengths - 3 + 2) / 2;
            let olens = 1 + (olens - 3 + 2) / 2;
            (olens - 1) / 2 + 1
        };
        // same arithmetic as AudioAdaptor::forward with k = 2
        let chunk_num = |seq_len: usize| (seq_len - 1) / 2 + 1;
        // realistic LFR'd feature lengths (lfr_n = 6 @ 10ms shift => ~17 frames per audio second)
        for speech_lengths in [17usize, 50, 100, 340, 1700] {
            assert!(
                fake_token_len(speech_lengths) <= chunk_num(speech_lengths),
                "speech_lengths {speech_lengths}: fake {} > adaptor chunks {}",
                fake_token_len(speech_lengths),
                chunk_num(speech_lengths)
            );
        }
    }

    /// masked_scatter_dim0 splices audio embeddings at fbank_mask positions and
    /// keeps the surrounding text embeddings untouched.
    #[test]
    fn masked_scatter_dim0_splice() {
        let device = Device::Cpu;
        let inputs_embeds = Tensor::zeros((1, 10, 8), DType::F32, &device).unwrap();
        let audio_embed = Tensor::ones((3, 8), DType::F32, &device).unwrap();
        let mut mask = vec![0u32; 10];
        mask[2..5].fill(1);
        let fbank_mask = Tensor::from_slice(&mask, (1, 10), &device).unwrap();
        let out = masked_scatter_dim0(&inputs_embeds, &audio_embed, &fbank_mask).unwrap();
        assert_eq!(out.dims(), &[1, 10, 8]);
        let out = out.squeeze(0).unwrap();
        for pos in 0..10 {
            let v: Vec<f32> = out.i(pos).unwrap().to_vec1().unwrap();
            let expected = if (2..5).contains(&pos) { 1.0 } else { 0.0 };
            assert!(v.iter().all(|&x| x == expected), "pos {pos}: {v:?}");
        }
    }
}
