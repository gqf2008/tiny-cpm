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

use crate::common::modules::sdpa_fast_guard;
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

/// Parse a boolean env knob: `=0`/`=off`/`=false`/`=no` (case-insensitive) disable,
/// `=1`/`=true` (or any other non-empty value) enable; unset falls back to
/// `default`. Replaces the old presence checks, where `QWEN3_TTS_CPU_SAMPLE=0`
/// still enabled the knob (presence, not value).
fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => false,
            _ => true,
        },
        Err(_) => default,
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
    layers: CodePredictorBackbone,
    norm: RmsNorm,
    rotary: RoPE,
    lm_head: Vec<Linear>, // 15 × [hidden → vocab]
    /// Weight dtype of the final norm / lm_head (BF16 in the checkpoint). The
    /// Quant backbone emits F32; `forward_hidden` casts back to this dtype
    /// (mirrors `Talker::dtype`, so a future full-F32 predictor works too).
    dtype: DType,
    /// talker_hidden → predictor_hidden projection. `None` when the two hidden sizes
    /// are equal (e.g. 0.6B, whose code predictor runs at the talker's 1024 and so has
    /// no `small_to_mtp_projection` tensor in the checkpoint); 1.7B projects 2048→1024.
    small_to_mtp: Option<Linear>,
}

/// The predictor's 5-layer Qwen3 stack: either the BF16 `Qwen3DecoderLayer`s or a
/// QMatMul-quantized mirror (same mechanism as the talker's `QuantizedTalkerBackbone`).
/// Quantizing the predictor cuts its per-frame weight-bandwidth floor ~3× (BF16
/// ~17ms → Q4_K ~5ms over the 15 sequential steps), the biggest RTF lever left.
enum CodePredictorBackbone {
    Full(Vec<Qwen3DecoderLayer>),
    Quant(quantized_talker::QuantizedTalkerBackbone),
}

impl CodePredictorBackbone {
    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            CodePredictorBackbone::Full(layers) => {
                let mut h = xs.clone();
                for l in layers {
                    h = l.forward(&h, cos, sin, mask)?;
                }
                Ok(h)
            }
            CodePredictorBackbone::Quant(backbone) => backbone.forward(xs, cos, sin, mask),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            CodePredictorBackbone::Full(layers) => {
                for l in layers {
                    l.clear_kv_cache();
                }
            }
            CodePredictorBackbone::Quant(backbone) => backbone.clear_kv_cache(),
        }
    }

    /// The Quant path emits F32 (QMatMul constraint); the Full path stays in the
    /// checkpoint's BF16. Used to cast the backbone output back before the BF16
    /// final norm / lm_head.
    fn is_quant(&self) -> bool {
        matches!(self, CodePredictorBackbone::Quant(_))
    }
}

impl CodePredictor {
    fn new(
        vb: VarBuilder,
        cfg: &CodePredictorConfig,
        talker_hidden: usize,
        device: &Device,
        vb_cpu: Option<&VarBuilder>,
        quant: Option<GgmlDType>,
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
        // Backbone: quantized QMatMul mirror when a CPU source + quant are given
        // (the live/perf path), else the BF16 Qwen3DecoderLayer stack.
        let layers = match (vb_cpu, quant) {
            (Some(cpu), Some(q)) => CodePredictorBackbone::Quant(quantized_talker::load_stack(
                cpu,
                "talker.code_predictor.model",
                cfg.num_hidden_layers,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.intermediate_size,
                cfg.rms_norm_eps,
                q,
                device,
            )?),
            _ => {
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
                let mut ls = Vec::with_capacity(cfg.num_hidden_layers);
                for i in 0..cfg.num_hidden_layers {
                    ls.push(Qwen3DecoderLayer::new(&qcfg, m.pp("layers").pp(i))?);
                }
                CodePredictorBackbone::Full(ls)
            }
        };
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
        // dtype of the norm/lm_head weights: the Quant backbone output is cast back
        // to this in `forward_hidden` (mirrors how `Talker::forward_step_gpu` casts
        // the quant talker output to `self.dtype`).
        let dtype = norm.weight().dtype();
        // 1.7B (talker 2048 → predictor 1024) carries a small_to_mtp_projection tensor;
        // 0.6B (both 1024) does not — skip the projection when dims already match.
        let small_to_mtp = if talker_hidden == cfg.hidden_size {
            None
        } else {
            Some(linear(
                talker_hidden,
                cfg.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        };
        Ok(Self {
            cfg: cfg.clone(),
            codec_embedding,
            layers,
            norm,
            rotary,
            lm_head,
            dtype,
            small_to_mtp,
        })
    }

    fn clear_kv_cache(&mut self) {
        self.layers.clear_kv_cache();
    }

    /// One decoder forward over `embeds` (1, T, talker_hidden), projecting to predictor
    /// hidden size first. Returns the last-position hidden (1, predictor_hidden).
    fn forward_hidden(&mut self, embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        // Project to predictor hidden only when the dims differ (0.6B runs them equal).
        let xs = match &self.small_to_mtp {
            Some(proj) => proj.forward(embeds)?,
            None => embeds.clone(),
        }; // (1, T, hidden)
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
        let h = self.layers.forward(&xs, &cos, &sin, mask.as_ref())?;
        // The Quant backbone returns F32; the final norm + lm_head run at
        // `self.dtype` (BF16 in the checkpoint), so cast back (mirrors how
        // Talker::forward_step_gpu casts the quant talker output to self.dtype).
        let h = if self.layers.is_quant() {
            h.to_dtype(self.dtype)?
        } else {
            h
        };
        let h = self.norm.forward(&h)?;
        Ok(h.narrow(1, seq_len - 1, 1)?) // (1, 1, hidden)
    }
}

/// Sample one token from a `(1, vocab)` logits row **entirely on-device** (no GPU→CPU
/// readback). Temperature is a scalar multiply and the multinomial draw is Gumbel-max
/// (`argmax(logits + Gumbel(0,1))`), which is exactly a softmax-weighted sample. Returns a
/// `(1,)` u32 tensor on the logits' device. top-k/top-p are skipped (see the note below).
///
/// `rep_mult`, when given, is a `(1, vocab)` repetition-penalty multiplier applied the HF
/// way: `logit < 0 ? logit * mult : logit / mult`. The talker keeps one on the GPU and
/// `scatter`s the penalty into the sampled column each frame, so codebook-0 history never
/// leaves the device (the code predictor passes `None` — it applies no penalty).
/// `#[doc(hidden)]`: exposed only so `tests/gpu_sampling.rs` can exercise the REAL
/// GPU sampling path (not a reimplementation); not part of the crate's API.
#[doc(hidden)]
pub fn gpu_sample_token(
    logits: &Tensor, // (1, vocab), any float dtype
    do_sample: bool,
    temperature: f64,
    top_k: usize,
    top_p: f32,
    rep_mult: Option<&Tensor>, // (1, vocab) f32, all-1 when no penalty
) -> Result<Tensor> {
    let mut lg = logits.to_dtype(DType::F32)?;

    // Repetition penalty (before temperature), HF formula applied elementwise:
    // negative logits are scaled up by mult, non-negative scaled down.
    if let Some(mult) = rep_mult {
        let mult = mult.to_dtype(DType::F32)?;
        let neg = lg.lt(0.0)?;
        let penalized_neg = (&lg * &mult)?;
        let penalized_pos = (&lg / &mult)?;
        lg = neg.where_cond(&penalized_neg, &penalized_pos)?;
    }

    // Temperature (skip when ~1 to avoid a no-op kernel).
    if do_sample && temperature > 0.0 && (temperature - 1.0).abs() > 1e-6 {
        lg = (lg / temperature)?;
    }

    // Top-k / top-p: skipped on the GPU path. The code predictor's defaults
    // (top_k=50, top_p=1.0 over a 2048-token vocab) only trim the negligible tail, and a
    // GPU top-k needs an O(vocab²) rank-count mask that costs far more than it saves on
    // Metal (measured: it made the predictor *slower* than the CPU-sync path). Gumbel-max
    // over the full temperature-scaled softmax is a faithful sample of essentially the
    // same distribution. (Set QWEN3_TTS_CPU_SAMPLE=1 for the exact top-k/top-p path.)
    let _ = top_k;
    let _ = top_p;

    if !do_sample {
        // Greedy: argmax over the (masked) logits.
        return Ok(lg.argmax(candle_core::D::Minus1)?); // (1,) u32
    }

    // Multinomial via Gumbel-max: sample u ~ U(0,1), g = -log(-log(u)), pick
    // argmax(lg + g). argmax over -inf entries stays -inf+finite → never selected.
    let u = Tensor::rand_like(&lg, 1e-7, 1.0)?; // avoid log(0)
    let gumbel = u.log()?.neg()?.log()?.neg()?;
    let perturbed = (lg + gumbel)?;
    Ok(perturbed.argmax(candle_core::D::Minus1)?) // (1,) u32
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
        Self::assemble(vb, tts, backbone, device, None, None)
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
        // Code-predictor quantization: clean back-to-back A/B shows Q4_K is a small
        // but real win (~25.1 vs ~26.1 ms/frame) — NOT the 3× the bandwidth math
        // hinted. The predictor is m=1 GEMV-occupancy-bound (GPU-busy, not launch
        // latency): fusing its QKV/gate-up projections cut the *launch* count but left
        // per-frame GPU-busy unchanged (~24ms, PROF_SYNC-measured), so it is NOT
        // launch-bound either. Default ON when the talker is quantized; opt out with
        // QWEN3_TTS_PREDICTOR_QUANT=0.
        let predictor_quant = match std::env::var("QWEN3_TTS_PREDICTOR_QUANT").as_deref() {
            Ok("0") | Ok("off") | Ok("false") | Ok("no") | Ok("none") => None,
            _ => Some(quant),
        };
        Self::assemble(vb, tts, backbone, device, Some(&vb_cpu), predictor_quant)
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
    /// `vb_cpu`+`predictor_quant`, when given, also quantize the code predictor's
    /// 5-layer stack (the live/perf path); `None` keeps it BF16.
    fn assemble(
        vb: VarBuilder,
        tts: &Qwen3TTSConfig,
        backbone: TalkerBackbone,
        device: &Device,
        vb_cpu: Option<&VarBuilder>,
        predictor_quant: Option<GgmlDType>,
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
            vb_cpu,
            predictor_quant,
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

    /// codec_embedding for one **on-device** token (1,1) u32 → (1, 1, hidden).
    /// Gather without a GPU→CPU readback (mlx-style lazy feedback).
    fn codec_embed1_gpu(&self, id_t: &Tensor) -> Result<Tensor> {
        self.codec_embedding.forward(id_t).map_err(Into::into)
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

    /// On-device variant of `frame_embed`: takes codebook-0 and the 15 predictor
    /// tokens as **on-device** (1,1) u32 tensors (no readback), sums their
    /// embeddings → (1, 1, hidden). Keeps the whole frame's token→embedding
    /// feedback on the GPU so the only per-frame sync is the single end-of-frame
    /// readback in `generate_stream`.
    fn frame_embed_gpu(&self, code0_t: &Tensor, rest_t: &[Tensor]) -> Result<Tensor> {
        let mut acc = self.codec_embed1_gpu(code0_t)?;
        for (g, tok) in rest_t.iter().enumerate() {
            let e = self.code_predictor.codec_embedding[g].forward(&tok.reshape((1, 1))?)?;
            acc = (acc + e)?;
        }
        Ok(acc)
    }

    /// One decoder forward over `embeds` (1, T, hidden), returning the last hidden
    /// (1,1,hidden) and the codec_head logits as an **on-device** (1, vocab) tensor.
    /// Keeping the logits on the GPU lets `generate_inner` sample codebook 0 without a
    /// full-vector readback (the per-frame `to_vec1` used to force a `flush_and_wait`
    /// that drained the whole pipeline).
    fn forward_step_gpu(
        &mut self,
        embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
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
        let logits = logits.squeeze(0)?; // (1, vocab)
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
                self.generate_inner(prompt, None, tts_pad_embed, gen_cfg, max_new_tokens, cb)
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
        // Fused SDPA decode fast path (`sdpa_vector_attention`) is opt-in: keep it
        // ON for this talker's decode forwards (both the Full and Quant backbones,
        // and the code predictor) and OFF everywhere else, so shared models
        // (qwen3-ASR, cosyvoice3, MOSS, fun_asr) keep the eager attention path.
        let _sdpa_fast = sdpa_fast_guard();
        // Copy config values out up front; `&mut self` methods are called in the loop.
        let vocab_size = self.cfg.vocab_size;
        let codec_eos = self.cfg.codec_eos_token_id;
        let num_code_groups = self.cfg.num_code_groups;
        let mut offset = prompt.dim(1)?;
        let (mut past_hidden, mut logits_gpu) = self.forward_step_gpu(&prompt, 0)?;
        let n_trailing = match &trailing {
            Some(t) => t.dim(1)?,
            None => 0,
        };

        let mut frames: Vec<Vec<u32>> = Vec::new();
        let suppress_from = vocab_size - 1024; // suppress [2048, 3072) except codec_eos
        // GPU suppression mask: (1, vocab) f32, -inf in [suppress_from, vocab) except codec_eos.
        // Precomputed once; added to the logits each frame before sampling codebook 0.
        let suppress_bias: Vec<f32> = (0..vocab_size)
            .map(|i| {
                if i >= suppress_from && i != codec_eos as usize {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let suppress_bias = Tensor::from_vec(suppress_bias, (1, vocab_size), &self.device)?;
        // Hoisted out of the frame loop: the logits dtype is stable across frames
        // (it is `self.dtype`), so the cast only needs to happen once.
        let suppress_bias = suppress_bias.to_dtype(logits_gpu.dtype())?;
        // GPU repetition-penalty multiplier (1, vocab), all-1 initially. After each frame
        // the sampled code0 column is scattered to `repetition_penalty`, so the penalty
        // compounds per distinct token exactly like the CPU path (HF applies it once per
        // distinct token; re-scattering the same value is idempotent).
        let rep_penalty = gen_cfg.repetition_penalty;
        let apply_rep = (rep_penalty - 1.0).abs() > 1e-6;
        let mut rep_mult = Tensor::ones((1, vocab_size), DType::F32, &self.device)?;
        // Scalar hoisted out of the frame loop (only used when apply_rep is true).
        let rep_pen = Tensor::from_vec(vec![rep_penalty], (1, 1), &self.device)?;
        // Diagnostic: QWEN3_TTS_GREEDY=1 forces argmax (do_sample=false) to test
        // whether babbling is sampling-driven (temperature × numeric scale).
        let greedy = env_flag("QWEN3_TTS_GREEDY", false);
        // QWEN3_TTS_PROF=1: per-stage wall-clock accumulation, printed at end of gen.
        let prof = env_flag("QWEN3_TTS_PROF", false);
        // QWEN3_TTS_PROF_SYNC=1: drain the GPU after each stage so each timer measures
        // that stage's true GPU completion time (not just CPU enqueue). Diagnostic only —
        // serializes the pipeline, so totals are pessimistic.
        let prof_sync = env_flag("QWEN3_TTS_PROF_SYNC", false);
        let drain = |dev: &Device| {
            if prof_sync {
                let _ = dev.synchronize();
            }
        };
        let mut t_sample0 = std::time::Duration::ZERO;
        let mut t_predictor = std::time::Duration::ZERO;
        let mut t_embed = std::time::Duration::ZERO;
        let mut t_fwd = std::time::Duration::ZERO;
        let mut t_cat = std::time::Duration::ZERO;
        let mut t_readback = std::time::Duration::ZERO;
        let t_gen = std::time::Instant::now();

        #[allow(clippy::explicit_counter_loop)]
        // `offset` tracks the KV-cache seqlen, not the loop item.
        for step in 0..max_new_tokens {
            let tt = std::time::Instant::now();
            // Sample codebook 0 on the GPU (suppression bias + repetition-penalty
            // multiplier + temperature + Gumbel-max), then keep the token ON-DEVICE.
            // mlx-audio alignment: defer the GPU→CPU sync — code0 feeds the predictor
            // and next-frame embedding as a lazy tensor, and the EOS flag is computed
            // on the GPU; the ONLY per-frame readback is the single cat at frame end.
            let lg = logits_gpu.broadcast_add(&suppress_bias)?;
            let rep = if apply_rep && step >= 2 {
                Some(&rep_mult)
            } else {
                None
            };
            let code0_t = gpu_sample_token(
                &lg,
                gen_cfg.do_sample && !greedy,
                gen_cfg.temperature,
                gen_cfg.top_k,
                gen_cfg.top_p,
                rep,
            )?; // (1,) u32 on device
            drain(&self.device);
            if prof {
                t_sample0 += tt.elapsed();
            }
            // Scatter the penalty into this token's multiplier column (idempotent per
            // distinct token — HF applies the penalty once per distinct token).
            if apply_rep {
                rep_mult = rep_mult.scatter(&code0_t.reshape((1, 1))?, &rep_pen, 1)?;
            }

            // code predictor fills codebooks 1..=15 (on-device tokens, no readback).
            let tt = std::time::Instant::now();
            let rest_t = self.code_predictor_predict(&past_hidden, &code0_t, gen_cfg)?;
            drain(&self.device);
            if prof {
                t_predictor += tt.elapsed();
            }

            // next input embedding: Σ16 codebook embeddings (on-device) + trailing/tts_pad.
            let tt = std::time::Instant::now();
            let mut next = self.frame_embed_gpu(&code0_t.reshape((1, 1))?, &rest_t)?;
            let text_row = match &trailing {
                Some(tr) if step < n_trailing => tr.narrow(1, step, 1)?,
                _ => tts_pad_embed.clone(),
            };
            next = (next + text_row)?;
            drain(&self.device);
            if prof {
                t_embed += tt.elapsed();
            }

            // SINGLE per-frame readback: cat [code0, rest..] into one (16,) tensor and
            // read it back once — for the EOS check, the streaming callback and the
            // returned codes tensor. (mlx-audio does the same: one mx.eval per frame.)
            let tt = std::time::Instant::now();
            let mut all_t = vec![code0_t.reshape((1,))?];
            all_t.extend(rest_t.iter().map(|t| t.reshape((1,)).unwrap()));
            let flat_c = Tensor::cat(&all_t, 0)?;
            if prof {
                t_cat += tt.elapsed();
            }
            let tt = std::time::Instant::now();
            let flat_frame = flat_c.to_vec1::<u32>()?;
            if prof {
                t_readback += tt.elapsed();
            }
            let code0 = flat_frame[0];
            let frame = flat_frame;

            if env_flag("QWEN3_TTS_DEBUG", false) && step % 25 == 0 {
                eprintln!("qwen3-tts: frame {step} code0={code0} (offset={offset})");
            }
            if env_flag("QWEN3_TTS_TRACE", false) {
                eprintln!("trace frame {step} code0={code0}");
            }

            // EOS check AFTER the predictor (matches mlx: is_eos.item() after the code
            // loop). The EOS frame's predictor codes are computed then dropped.
            if code0 == codec_eos && step >= 2 {
                break;
            }
            frames.push(frame.clone());

            // Streaming: emit this frame's codes; `false` aborts generation early.
            if !on_frame(&frames[frames.len() - 1]) {
                break;
            }

            let tt = std::time::Instant::now();
            let (h, l) = self.forward_step_gpu(&next, offset)?;
            drain(&self.device);
            if prof {
                t_fwd += tt.elapsed();
            }
            past_hidden = h;
            logits_gpu = l;
            offset += 1;
        }

        if prof {
            let total = t_gen.elapsed();
            eprintln!(
                "qwen3-tts PROF: frames={} total={:.2?} | sample0={:.2?} predictor={:.2?} embed={:.2?} fwd={:.2?} cat={:.2?} readback={:.2?} | per-frame: sample0={:.2?} predictor={:.2?} embed={:.2?} fwd={:.2?} cat={:.2?} readback={:.2?}",
                frames.len(),
                total,
                t_sample0,
                t_predictor,
                t_embed,
                t_fwd,
                t_cat,
                t_readback,
                t_sample0 / frames.len().max(1) as u32,
                t_predictor / frames.len().max(1) as u32,
                t_embed / frames.len().max(1) as u32,
                t_fwd / frames.len().max(1) as u32,
                t_cat / frames.len().max(1) as u32,
                t_readback / frames.len().max(1) as u32,
            );
        }

        let flat: Vec<u32> = frames.into_iter().flatten().collect();
        let n = flat.len() / num_code_groups;
        if std::env::var("QWEN3_TTS_DUMP_CODES").is_ok() {
            eprintln!("CANDLE_CODES frames={n} {flat:?}");
        }
        Ok(Tensor::from_vec(flat, (n, num_code_groups), &self.device)?)
    }

    /// Run the code predictor for one frame (codebook 0 already known). Returns 15 codes.
    /// Run the code predictor for one frame (codebook 0 already known). Returns the
    /// 15 codebook-1..=15 tokens as **on-device** (1,) u32 tensors (default GPU path)
    /// so the caller can feed them back into `frame_embed_gpu` without a readback;
    /// the single per-frame readback happens in `generate_stream`.
    ///
    /// **GPU-sampling path (default)** — the 15 codebook steps each need the sampled token
    /// to build the next step's embedding, which naively forces a blocking `to_vec2`
    /// GPU→CPU readback per step. Profiling (`QWEN3_TTS_PROF=1`) shows that readback is
    /// the dominant per-frame cost: it drains the entire queued GPU command buffer, so
    /// 15 serialized syncs prevent any pipelining and the predictor alone was ~70% of
    /// frame time. The fix keeps sampling *on the GPU*: each step applies temperature +
    /// top-k + Gumbel-max argmax as tensor ops (no readback) and feeds the token back via
    /// an on-device embedding gather. Set `QWEN3_TTS_CPU_SAMPLE=1` to restore the
    /// per-step CPU sampling (reference path, returns read-back values).
    fn code_predictor_predict(
        &mut self,
        talker_hidden_last: &Tensor,
        code0_t: &Tensor,
        gen_cfg: &Qwen3TTSGenerationConfig,
    ) -> Result<Vec<Tensor>> {
        let code0_emb = self.codec_embed1_gpu(&code0_t.reshape((1, 1))?)?; // talker codec_embedding for book 0
        let cp = &mut self.code_predictor;
        cp.clear_kv_cache();
        let prefill = Tensor::cat(&[talker_hidden_last.clone(), code0_emb], 1)?; // (1,2,hidden)
        let mut hidden = cp.forward_hidden(&prefill, 0)?; // (1,1,hidden) from position 1
        let n_groups = cp.cfg.num_code_groups - 1;
        let mut offset = 2usize;
        let cpu_sample = env_flag("QWEN3_TTS_CPU_SAMPLE", false);

        if cpu_sample {
            // Reference path: per-step CPU sampling (one blocking sync per codebook).
            // Returns the tokens re-uploaded as (1,) u32 tensors to match the signature.
            let mut codes = Vec::with_capacity(n_groups);
            for g in 0..n_groups {
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
                if g + 1 < n_groups {
                    let emb = cp.codec_embedding[g].forward(&Tensor::from_vec(
                        vec![token],
                        (1, 1),
                        hidden.device(),
                    )?)?; // (1,1,talker_hidden)
                    hidden = cp.forward_hidden(&emb, offset)?;
                    offset += 1;
                }
                codes.push(Tensor::from_vec(vec![token], (1,), &self.device)?);
            }
            return Ok(codes);
        }

        // GPU path: keep the running token as an on-device (1,) u32 tensor; gather its
        // embedding without a readback. Collect the 15 token tensors; the caller does
        // the single end-of-frame readback (code0 + these + EOS) in generate_stream.
        let spec_probe = env_flag("QWEN3_TTS_SPEC_PROBE", false);
        let mut token_tensors: Vec<Tensor> = Vec::with_capacity(n_groups);
        for g in 0..n_groups {
            let logits = cp.lm_head[g].forward(&hidden)?.squeeze(0)?; // (1, vocab)
            // Speculatability probe (greedy/argmax only, read-only): how confident is the
            // target at this step, and would cheap drafts (copy-prev-book) have hit?
            if spec_probe {
                let lf: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?[0].clone();
                let mut idx: Vec<usize> = (0..lf.len()).collect();
                idx.sort_by(|&a, &b| lf[b].total_cmp(&lf[a]));
                let (t1, t2) = (lf[idx[0]], lf[idx[1]]);
                let max = lf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let p1 = (lf[idx[0]] - max).exp();
                let z: f32 = lf.iter().map(|&x| (x - max).exp()).sum();
                eprintln!(
                    "spec g={:2} top1_id={} top1_logit={:.2} margin={:.2} p1={:.3}",
                    g,
                    idx[0],
                    t1,
                    t1 - t2,
                    p1 / z
                );
            }
            if std::env::var("QWEN3_TTS_DUMP_CODES").is_ok() && g == 6 {
                let lf: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?[0].clone();
                let mut idx: Vec<usize> = (0..lf.len()).collect();
                idx.sort_by(|&a, &b| lf[b].total_cmp(&lf[a]));
                eprintln!(
                    "CANDLE_DUMP cp g=0 top5 {:?}",
                    idx[..5].iter().map(|&i| (lf[i], i)).collect::<Vec<_>>()
                );
            }
            let token = gpu_sample_token(
                &logits,
                gen_cfg.subtalker_dosample && std::env::var("QWEN3_TTS_GREEDY").is_err(),
                gen_cfg.subtalker_temperature,
                gen_cfg.subtalker_top_k,
                gen_cfg.subtalker_top_p,
                None, // code predictor applies no repetition penalty
            )?; // (1,) u32 on device
            if g + 1 < n_groups {
                let emb = cp.codec_embedding[g].forward(&token.reshape((1, 1))?)?;
                hidden = cp.forward_hidden(&emb, offset)?;
                offset += 1;
            }
            token_tensors.push(token);
        }
        Ok(token_tensors)
    }
}
