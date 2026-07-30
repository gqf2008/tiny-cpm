//! Qwen3-TTS talker (LM) + code predictor, ported from
//! github.com/QwenLM/Qwen3-TTS qwen_tts/core/models/modeling_qwen3_tts.py
//! (`Qwen3TTSTalkerForConditionalGeneration`, `Qwen3TTSTalkerModel`,
//! `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`,
//! `Qwen3TTSForConditionalGeneration.generate{,_icl_prompt,_speaker_prompt}`).
//!
//! The talker is a stock Qwen3 decoder (reused `Qwen3DecoderLayer`: RMSNorm, RoPE
//! θ1e6, GQA, per-head Q/K RMSNorm, SwiGLU) with a **dual-track embedding**:
//! `text_embedding`[text_vocab×text_hidden] → `text_projection` (Linear→SiLU→Linear,
//! → hidden) summed with `codec_embedding`[vocab×hidden]. It emits codec codebook 0
//! via `codec_head` per 12.5 Hz frame; the 5-layer **code predictor** then fills
//! codebooks 1..=15 autoregressively (`lm_head[0..14]`, `codec_embedding[0..14]`,
//! `small_to_mtp_projection` maps talker hidden → predictor hidden).
//!
//! M-RoPE is a no-op here (`get_rope_index` returns 3 identical rows for a pure
//! 1-D sequence) → plain RoPE with `seqlen_offset` tracking. Voice cloning is the
//! **ICL continuation** layout (see `Talker::generate`): speaker embedding as one
//! prefix row + ref codec codes prepended to the prompt; per generated frame the
//! text track adds the next "trailing text" token (or `tts_pad` once text runs out).
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    Activation, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear, linear_no_bias,
    rms_norm,
};

use crate::common::sample::sample_from_logits_vec;
use crate::models::qwen3::config::Qwen3Config;
use crate::models::qwen3::model::Qwen3DecoderLayer;
use crate::models::qwen3_tts::config::{
    CodePredictorConfig, Qwen3TTSConfig, Qwen3TTSGenerationConfig, TalkerConfig,
};
use crate::models::qwen3_tts::quantized_talker::{self, QuantizedTalkerBackbone};
use crate::position_embed::rope::RoPE;
use crate::utils::tensor_utils::prepare_causal_attention_mask;
use candle_core::quantized::GgmlDType;

/// Build a `Qwen3Config` for the reused `Qwen3DecoderLayer` (only the fields the
/// layer reads matter; the rest are inert defaults).
#[allow(clippy::too_many_arguments)]
fn qwen3_cfg(
    hidden: usize,
    inter: usize,
    layers: usize,
    heads: usize,
    kv: usize,
    head_dim: usize,
    eps: f64,
    theta: f64,
    act: &str,
    vocab: usize,
) -> Qwen3Config {
    Qwen3Config {
        attention_bias: false,
        attention_dropout: 0.0,
        bos_token_id: 0,
        eos_token_id: 0,
        head_dim,
        hidden_act: match act {
            "gelu" | "gelu_new" => Activation::NewGelu,
            _ => Activation::Silu, // qwen3-tts uses silu everywhere
        },
        hidden_size: hidden,
        initializer_range: 0.02,
        intermediate_size: inter,
        max_position_embeddings: 32768,
        max_window_layers: layers,
        num_attention_heads: heads,
        num_hidden_layers: layers,
        num_key_value_heads: kv,
        rms_norm_eps: eps,
        rope_theta: theta as f32,
        tie_word_embeddings: false,
        torch_dtype: "bfloat16".into(),
        use_cache: true,
        use_sliding_window: false,
        vocab_size: vocab,
    }
}

/// `Qwen3TTSTalkerResizeMLP`: Linear(text_hidden→text_hidden) → SiLU → Linear(→hidden).
/// (Both matmuls are 2048→2048 for this checkpoint.)
struct ResizeMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ResizeMlp {
    fn new(vb: VarBuilder, text_hidden: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            fc1: linear(text_hidden, text_hidden, vb.pp("linear_fc1"))?,
            fc2: linear(text_hidden, hidden, vb.pp("linear_fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.fc2.forward(&self.fc1.forward(x)?.silu()?)?)
    }
}

/// The 5-layer code predictor (`talker.code_predictor.*`): predicts codebooks 1..=15.
struct CodePredictor {
    cfg: CodePredictorConfig,
    codec_embedding: Vec<Embedding>, // 15 × [vocab × talker_hidden]
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    rotary: RoPE,
    lm_head: Vec<Linear>, // 15 × [hidden → vocab]
    small_to_mtp: Linear, // talker_hidden → predictor hidden
}

impl CodePredictor {
    fn new(
        vb: VarBuilder,
        cfg: &CodePredictorConfig,
        talker_hidden: usize,
        device: &Device,
    ) -> Result<Self> {
        let m = vb.pp("model");
        let mut codec_embedding = Vec::with_capacity(cfg.num_code_groups - 1);
        for g in 0..cfg.num_code_groups - 1 {
            codec_embedding.push(embedding(
                cfg.vocab_size,
                talker_hidden,
                m.pp("codec_embedding").pp(g),
            )?);
        }
        let qcfg = qwen3_cfg(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rms_norm_eps,
            cfg.rope_theta,
            &cfg.hidden_act,
            cfg.vocab_size,
        );
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(&qcfg, m.pp("layers").pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, m.pp("norm"))?;
        let rotary = RoPE::new(cfg.head_dim, cfg.rope_theta as f32, device)?;
        let mut lm_head = Vec::with_capacity(cfg.num_code_groups - 1);
        for g in 0..cfg.num_code_groups - 1 {
            lm_head.push(linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head").pp(g),
            )?);
        }
        let small_to_mtp = linear(
            talker_hidden,
            cfg.hidden_size,
            vb.pp("small_to_mtp_projection"),
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            codec_embedding,
            layers,
            norm,
            rotary,
            lm_head,
            small_to_mtp,
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    /// One decoder forward over `embeds` (1, T, talker_hidden), projecting to predictor
    /// hidden size first. Returns the last-position hidden (1, predictor_hidden).
    fn forward_hidden(&mut self, embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let xs = self.small_to_mtp.forward(embeds)?; // (1, T, hidden)
        let (bs, seq_len, _) = xs.dims3()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(prepare_causal_attention_mask(
                bs,
                seq_len,
                seqlen_offset,
                xs.device(),
            )?)
        };
        let (cos, sin) = self.rotary.forward(seqlen_offset, seq_len, xs.device())?;
        let mut h = xs;
        for l in &mut self.layers {
            h = l.forward(&h, &cos, &sin, mask.as_ref())?;
        }
        h = self.norm.forward(&h)?;
        Ok(h.narrow(1, seq_len - 1, 1)?) // (1, 1, hidden)
    }
}

/// Voice-cloning reference: pre-encoded speaker embedding + ref codec codes + ref text.
pub struct RefVoice {
    /// Raw speaker embedding, (enc_dim,) — becomes one prefix row.
    pub spk_embed: Tensor,
    /// Ref codec codes, (16, T_ref) u32.
    pub ref_code: Tensor,
    /// Ref transcript token ids (the ref clip's text), for the ICL text track.
    pub ref_text_ids: Vec<u32>,
}

/// Talker backbone dispatch: the BF16 `Qwen3DecoderLayer` stack, or the QMatMul
/// Q4_K mirror (`quantized_talker`). Same forward/clear_kv_cache shape; the
/// Quant arm runs F32 activations (Metal QMatMul constraint) and its output is
/// cast back to the talker dtype by the caller.
enum TalkerBackbone {
    Full(Vec<Qwen3DecoderLayer>),
    Quant(QuantizedTalkerBackbone),
}

impl TalkerBackbone {
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::Full(layers) => {
                let mut h = xs.clone();
                for l in layers.iter_mut() {
                    h = l.forward(&h, cos, sin, attention_mask)?;
                }
                Ok(h)
            }
            Self::Quant(b) => b.forward(xs, cos, sin, attention_mask),
        }
    }
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Full(layers) => {
                for l in layers.iter_mut() {
                    l.clear_kv_cache();
                }
            }
            Self::Quant(b) => b.clear_kv_cache(),
        }
    }

    /// Diagnostic: run the stack returning each layer's output (F32), to locate
    /// the first layer that diverges from a reference trajectory.
    fn forward_trace(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        match self {
            Self::Full(layers) => {
                // Keep the reference dtype end-to-end; cast each output to F32
                // only for the cross-backbone comparison.
                let mut h = xs.clone();
                let mut outs = Vec::with_capacity(layers.len());
                for l in layers.iter_mut() {
                    h = l.forward(&h, cos, sin, attention_mask)?;
                    outs.push(h.to_dtype(DType::F32)?);
                }
                Ok(outs)
            }
            Self::Quant(b) => b.forward_trace(xs, cos, sin, attention_mask),
        }
    }
}

pub struct Talker {
    cfg: TalkerConfig,
    tts: Qwen3TTSConfig,
    device: Device,
    text_embedding: Embedding,  // [text_vocab × text_hidden]
    text_projection: ResizeMlp, // text_hidden → hidden
    codec_embedding: Embedding, // [vocab × hidden]
    backbone: TalkerBackbone,
    norm: RmsNorm,
    rotary: RoPE,
    codec_head: Linear, // [hidden → vocab]
    /// dtype the non-quantized weights run in (BF16); the Quant backbone outputs
    /// F32 and is cast to this before `norm`/`codec_head`.
    dtype: DType,
    code_predictor: CodePredictor,
}

impl Talker {
    pub fn new(vb: VarBuilder, tts: &Qwen3TTSConfig, device: &Device) -> Result<Self> {
        let cfg = &tts.talker_config;
        let qcfg = Self::layer_cfg(cfg);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(
                &qcfg,
                vb.pp("talker").pp("model").pp("layers").pp(i),
            )?);
        }
        let backbone = TalkerBackbone::Full(layers);
        Self::assemble(vb, tts, backbone, device)
    }

    /// Quantized backbone (QMatMul, e.g. Q4_K). The 7 big matmuls per layer are
    /// quantized in-memory from a **CPU-mmaped** `vb_cpu`; all other weights
    /// (embeddings / text_projection / codec_head / predictor) load from `vb`
    /// as in `new`.
    pub fn new_quantized(
        vb: VarBuilder,
        vb_cpu: VarBuilder,
        tts: &Qwen3TTSConfig,
        quant: GgmlDType,
        device: &Device,
    ) -> Result<Self> {
        let cfg = &tts.talker_config;
        // vb_cpu is rooted at the repo root and mmaped on CPU; quantized_talker
        // prefixes `talker.model.` itself and quantizes onto `device`.
        let backbone = TalkerBackbone::Quant(quantized_talker::load(&vb_cpu, cfg, quant, device)?);
        Self::assemble(vb, tts, backbone, device)
    }

    /// The `Qwen3Config` passed to the reused `Qwen3DecoderLayer` (Full path).
    fn layer_cfg(cfg: &TalkerConfig) -> Qwen3Config {
        qwen3_cfg(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rms_norm_eps,
            cfg.rope_theta,
            &cfg.hidden_act,
            cfg.vocab_size,
        )
    }

    /// Shared constructor: load every non-backbone weight and assemble the talker.
    fn assemble(
        vb: VarBuilder,
        tts: &Qwen3TTSConfig,
        backbone: TalkerBackbone,
        device: &Device,
    ) -> Result<Self> {
        let cfg = &tts.talker_config;
        let vb_t = vb.pp("talker");
        let m = vb_t.pp("model");
        let text_embedding = embedding(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            m.pp("text_embedding"),
        )?;
        let text_projection = ResizeMlp::new(
            vb_t.pp("text_projection"),
            cfg.text_hidden_size,
            cfg.hidden_size,
        )?;
        let codec_embedding = embedding(cfg.vocab_size, cfg.hidden_size, m.pp("codec_embedding"))?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, m.pp("norm"))?;
        let rotary = RoPE::new(cfg.head_dim, cfg.rope_theta as f32, device)?;
        let codec_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_t.pp("codec_head"))?;
        let dtype = codec_head.weight().dtype();
        let code_predictor = CodePredictor::new(
            vb_t.pp("code_predictor"),
            &cfg.code_predictor_config,
            cfg.hidden_size,
            device,
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            tts: tts.clone(),
            device: device.clone(),
            text_embedding,
            text_projection,
            codec_embedding,
            backbone,
            norm,
            rotary,
            codec_head,
            dtype,
            code_predictor,
        })
    }

    fn clear_kv_cache(&mut self) {
        self.backbone.clear_kv_cache();
    }

    // --- embedding helpers -------------------------------------------------

    /// text_embedding + text_projection for a row of token ids → (1, N, hidden).
    fn text_embed(&self, ids: &[u32]) -> Result<Tensor> {
        let t = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)?;
        let emb = self.text_embedding.forward(&t)?;
        self.text_projection.forward(&emb)
    }

    /// codec_embedding for one id → (1, 1, hidden).
    fn codec_embed1(&self, id: u32) -> Result<Tensor> {
        let t = Tensor::from_vec(vec![id], (1, 1), &self.device)?;
        self.codec_embedding.forward(&t).map_err(Into::into)
    }

    /// Sum of all 16 codebook embeddings for one frame's codes → (1, 1, hidden).
    /// codes[0] uses the talker codec_embedding; codes[1..16] use the predictor's.
    fn frame_embed(&self, codes: &[u32]) -> Result<Tensor> {
        let mut acc = self.codec_embed1(codes[0])?;
        for g in 1..self.cfg.num_code_groups {
            let e = self.code_predictor.codec_embedding[g - 1].forward(&Tensor::from_vec(
                vec![codes[g]],
                (1, 1),
                &self.device,
            )?)?;
            acc = (acc + e)?;
        }
        Ok(acc)
    }

    /// One decoder forward over `embeds` (1, T, hidden) → (last hidden (1,1,hidden),
    /// codec_head logits (vocab,) f32).
    fn forward_step(
        &mut self,
        embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Vec<f32>)> {
        let (bs, seq_len, _) = embeds.dims3()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(prepare_causal_attention_mask(
                bs,
                seq_len,
                seqlen_offset,
                embeds.device(),
            )?)
        };
        let (cos, sin) = self
            .rotary
            .forward(seqlen_offset, seq_len, embeds.device())?;
        let h = self.backbone.forward(embeds, &cos, &sin, mask.as_ref())?;
        // The Quant backbone runs F32 (Metal QMatMul); cast back to the talker
        // dtype for norm/codec_head. No-op on the Full path.
        let h = h.to_dtype(self.dtype)?;
        let h = self.norm.forward(&h)?;
        let last = h.narrow(1, seq_len - 1, 1)?; // (1,1,hidden)
        let logits = self.codec_head.forward(&last)?; // (1,1,vocab)
        let logits = logits
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        if std::env::var("QWEN3_TTS_NUMDBG").is_ok() {
            let lv: Vec<f32> = last.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            eprintln!(
                "numdbg offset={seqlen_offset} seq={seq_len} last[0..4]={:?} top3={:?}",
                &lv[..lv.len().min(4)],
                &idx[..3].iter().map(|&i| (i, logits[i])).collect::<Vec<_>>()
            );
        }
        Ok((last, logits))
    }

    /// Generate codec frames for `text_ids` (target text, tokenized by the exec driver)
    /// with optional ICL voice cloning. `newline_token` is the BPE id for '\n' used in
    /// the "<|im_start|>assistant\n" role prefix. Returns codes (n_frames, 16) u32.
    pub fn generate(
        &mut self,
        text_ids: &[u32],
        language: &str,
        ref_voice: Option<&RefVoice>,
        newline_token: u32,
        gen_cfg: &Qwen3TTSGenerationConfig,
        max_new_tokens: usize,
    ) -> Result<Tensor> {
        self.generate_stream(
            text_ids,
            language,
            ref_voice,
            newline_token,
            gen_cfg,
            max_new_tokens,
            None,
        )
    }

    /// `generate` + an optional per-frame callback. After each frame's 16 codes are
    /// finalized, `on_frame(&frame)` runs; returning `false` aborts generation early
    /// (the frames so far are still returned, so the caller can flush the tail). The
    /// exec streaming path uses this to incrementally codec-decode as frames arrive.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream(
        &mut self,
        text_ids: &[u32],
        language: &str,
        ref_voice: Option<&RefVoice>,
        newline_token: u32,
        gen_cfg: &Qwen3TTSGenerationConfig,
        max_new_tokens: usize,
        mut on_frame: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<Tensor> {
        anyhow::ensure!(!text_ids.is_empty(), "talker.generate: empty text");
        self.clear_kv_cache();
        // Normalize to a single &mut dyn callback (no-op when streaming is off) so the
        // ICL / non-ICL branches below can pass one uniform reference to generate_inner.
        let mut noop = |_: &[u32]| true;
        let cb: &mut dyn FnMut(&[u32]) -> bool = match on_frame.as_mut() {
            Some(c) => &mut **c,
            None => &mut noop,
        };
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;

        // --- special embeds (text-projected bos/eos/pad) -------------------
        let bep = self.text_embed(&[
            self.tts.tts_bos_token_id,
            self.tts.tts_eos_token_id,
            self.tts.tts_pad_token_id,
        ])?; // (1,3,hidden)
        let tts_bos_embed = bep.narrow(1, 0, 1)?;
        let tts_eos_embed = bep.narrow(1, 1, 1)?;
        let tts_pad_embed = bep.narrow(1, 2, 1)?;

        // --- codec-track prefix: think/nothink block + (spk) + [pad, bos] ---
        let language_id = if language.eq_ignore_ascii_case("auto") {
            None
        } else {
            Some(
                *cfg.codec_language_id
                    .get(&language.to_lowercase())
                    .ok_or_else(|| anyhow!("language {language} not in codec_language_id"))?,
            )
        };
        let think_ids: Vec<u32> = match language_id {
            None => vec![
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
            ],
            Some(lang) => vec![
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                lang,
                cfg.codec_think_eos_id,
            ],
        };
        let mut codec_rows: Vec<Tensor> = Vec::new();
        for &id in &think_ids {
            codec_rows.push(self.codec_embed1(id)?);
        }
        if let Some(rv) = ref_voice {
            codec_rows.push(rv.spk_embed.reshape((1, 1, hidden))?);
        }
        codec_rows.push(self.codec_embed1(cfg.codec_pad_id)?);
        codec_rows.push(self.codec_embed1(cfg.codec_bos_id)?);
        let codec_prefix = Tensor::cat(&codec_rows, 1)?; // (1, P, hidden)
        let p_len = codec_prefix.dim(1)?;

        // --- role prefix <|im_start|>assistant\n + (pads+bos text) ⊕ codec prefix ---
        let role = self.text_embed(&[
            self.tts.im_start_token_id,
            self.tts.assistant_token_id,
            newline_token,
        ])?; // (1,3,hidden)
        let n_pad = p_len - 2;
        let pad_rows = if n_pad > 0 {
            tts_pad_embed.broadcast_as((1, n_pad, hidden))?
        } else {
            tts_pad_embed.narrow(1, 0, 0)?
        };
        let text_track_head = Tensor::cat(&[pad_rows, tts_bos_embed], 1)?; // (1, P-1, hidden)
        let head = (text_track_head + codec_prefix.narrow(1, 0, p_len - 1)?)?; // (1, P-1, hidden)
        let prompt0 = Tensor::cat(&[role, head], 1)?; // (1, 3 + (P-1), hidden)

        // --- text body + ICL ------------------------------------------------
        if let Some(rv) = ref_voice {
            // ICL: text track = ref_text + target_text + eos; codec track = ref codes.
            let mut text_body_ids: Vec<u32> = rv.ref_text_ids.clone();
            text_body_ids.extend_from_slice(text_ids);
            let text_body = self.text_embed(&text_body_ids)?;
            let text_track = Tensor::cat(&[text_body, tts_eos_embed], 1)?; // (1, T1+1, hidden)
            // codec track: [codec_bos] + Σ16 embeddings per ref frame.
            let ref_code = &rv.ref_code; // (16, T_ref)
            let t_ref = ref_code.dim(1)?;
            let mut codec_track_rows = vec![self.codec_embed1(cfg.codec_bos_id)?];
            for fr in 0..t_ref {
                let codes: Vec<u32> = ref_code.narrow(1, fr, 1)?.squeeze(1)?.to_vec1::<u32>()?;
                codec_track_rows.push(self.frame_embed(&codes)?);
            }
            let codec_track = Tensor::cat(&codec_track_rows, 1)?; // (1, T_ref+1, hidden)
            let t1 = text_track.dim(1)?;
            let c1 = codec_track.dim(1)?;
            if t1 >= c1 {
                let head_part = (text_track.narrow(1, 0, c1)? + codec_track)?;
                let prompt = Tensor::cat(&[prompt0, head_part], 1)?;
                let rest = text_track.narrow(1, c1, t1 - c1)?; // trailing
                self.generate_inner(
                    prompt,
                    Some(rest),
                    tts_pad_embed,
                    gen_cfg,
                    max_new_tokens,
                    cb,
                )
            } else {
                let pad_rows = Tensor::cat(&vec![tts_pad_embed.clone(); c1 - t1], 1)?;
                let text_track = Tensor::cat(&[text_track, pad_rows], 1)?;
                let head_part = (text_track + codec_track)?;
                let prompt = Tensor::cat(&[prompt0, head_part], 1)?;
                self.generate_inner(
                    prompt,
                    None,
                    tts_pad_embed,
                    gen_cfg,
                    max_new_tokens,
                    cb,
                )
            }
        } else {
            // Non-ICL: first text token ⊕ last codec prefix row (bos); rest → trailing.
            let first =
                (self.text_embed(&text_ids[0..1])? + codec_prefix.narrow(1, p_len - 1, 1)?)?;
            let prompt = Tensor::cat(&[prompt0, first], 1)?;
            let rest = if text_ids.len() > 1 {
                let body = self.text_embed(&text_ids[1..])?;
                Tensor::cat(&[body, tts_eos_embed], 1)?
            } else {
                tts_eos_embed
            };
            self.generate_inner(
                prompt,
                Some(rest),
                tts_pad_embed,
                gen_cfg,
                max_new_tokens,
                cb,
            )
        }
    }

    /// The AR decode loop shared by ICL and non-ICL. `trailing` rows (1, N, hidden) are
    /// added to the text track one per generated frame; once exhausted, `tts_pad_embed`.
    /// `on_frame` (streaming) runs after each frame's 16 codes are finalized; `false`
    /// aborts the loop early (frames generated so far are kept and returned).
    fn generate_inner(
        &mut self,
        prompt: Tensor,
        trailing: Option<Tensor>,
        tts_pad_embed: Tensor,
        gen_cfg: &Qwen3TTSGenerationConfig,
        max_new_tokens: usize,
        on_frame: &mut dyn FnMut(&[u32]) -> bool,
    ) -> Result<Tensor> {
        // Copy config values out up front; `&mut self` methods are called in the loop.
        let vocab_size = self.cfg.vocab_size;
        let codec_eos = self.cfg.codec_eos_token_id;
        let num_code_groups = self.cfg.num_code_groups;
        let mut offset = prompt.dim(1)?;
        let (mut past_hidden, mut logits) = self.forward_step(&prompt, 0)?;
        let n_trailing = match &trailing {
            Some(t) => t.dim(1)?,
            None => 0,
        };

        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut gen_history: Vec<u32> = Vec::new(); // codebook-0 history for rep penalty
        let suppress_from = vocab_size - 1024; // suppress [2048, 3072) except codec_eos
        // Diagnostic: QWEN3_TTS_GREEDY=1 forces argmax (do_sample=false) to test
        // whether babbling is sampling-driven (temperature × numeric scale).
        let greedy = std::env::var("QWEN3_TTS_GREEDY").is_ok();

        #[allow(clippy::explicit_counter_loop)]
        // `offset` tracks the KV-cache seqlen, not the loop item.
        for step in 0..max_new_tokens {
            let mut lg = logits.clone();
            for (i, l) in lg.iter_mut().enumerate().take(vocab_size) {
                if i >= suppress_from && i != codec_eos as usize {
                    *l = f32::NEG_INFINITY;
                }
            }
            let code0 = sample_from_logits_vec(
                &lg,
                gen_cfg.do_sample && !greedy,
                Some(gen_cfg.temperature),
                Some(gen_cfg.top_k),
                Some(gen_cfg.top_p),
                if step >= 2 { Some(&gen_history) } else { None },
                gen_cfg.repetition_penalty,
            )?;
            if code0 == codec_eos && step >= 2 {
                break;
            }
            gen_history.push(code0);
            if std::env::var("QWEN3_TTS_DEBUG").is_ok() && step % 25 == 0 {
                eprintln!("qwen3-tts: frame {step} code0={code0} (offset={offset})");
            }
            if std::env::var("QWEN3_TTS_TRACE").is_ok() {
                eprintln!("trace frame {step} code0={code0}");
            }

            // code predictor fills codebooks 1..=15.
            let rest = self.code_predictor_predict(&past_hidden, code0, gen_cfg)?;
            let mut frame = vec![code0];
            frame.extend_from_slice(&rest);

            // next input embedding: Σ16 codebook embeddings + trailing/tts_pad text row.
            let mut next = self.frame_embed(&frame)?;
            let text_row = match &trailing {
                Some(tr) if step < n_trailing => tr.narrow(1, step, 1)?,
                _ => tts_pad_embed.clone(),
            };
            next = (next + text_row)?;
            frames.push(frame);

            // Streaming: emit this frame's codes; `false` aborts generation early.
            if !on_frame(&frames[frames.len() - 1]) {
                break;
            }

            let (h, l) = self.forward_step(&next, offset)?;
            past_hidden = h;
            logits = l;
            offset += 1;
        }

        let flat: Vec<u32> = frames.into_iter().flatten().collect();
        let n = flat.len() / num_code_groups;
        Ok(Tensor::from_vec(flat, (n, num_code_groups), &self.device)?)
    }

    /// Run the code predictor for one frame (codebook 0 already known). Returns 15 codes.
    fn code_predictor_predict(
        &mut self,
        talker_hidden_last: &Tensor,
        code0: u32,
        gen_cfg: &Qwen3TTSGenerationConfig,
    ) -> Result<Vec<u32>> {
        let code0_emb = self.codec_embed1(code0)?; // talker codec_embedding for book 0
        let cp = &mut self.code_predictor;
        cp.clear_kv_cache();
        let prefill = Tensor::cat(&[talker_hidden_last.clone(), code0_emb], 1)?; // (1,2,hidden)
        let mut hidden = cp.forward_hidden(&prefill, 0)?; // (1,1,hidden) from position 1
        let mut codes = Vec::with_capacity(cp.cfg.num_code_groups - 1);
        let mut offset = 2usize;
        for g in 0..cp.cfg.num_code_groups - 1 {
            let logits = cp.lm_head[g]
                .forward(&hidden)?
                .squeeze(1)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let token = sample_from_logits_vec(
                &logits[0],
                gen_cfg.subtalker_dosample,
                Some(gen_cfg.subtalker_temperature),
                Some(gen_cfg.subtalker_top_k),
                Some(gen_cfg.subtalker_top_p),
                None,
                1.0,
            )?;
            codes.push(token);
            if g + 1 < cp.cfg.num_code_groups - 1 {
                let emb = cp.codec_embedding[g].forward(&Tensor::from_vec(
                    vec![token],
                    (1, 1),
                    hidden.device(),
                )?)?; // (1,1,talker_hidden)
                hidden = cp.forward_hidden(&emb, offset)?;
                offset += 1;
            }
        }
        Ok(codes)
    }
}
