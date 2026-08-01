//! Qwen3-TTS talker (LM) + code predictor on burn, ported 1:1 from tiny-cpm
//! src/models/qwen3_tts/talker.rs (which ports QwenLM/Qwen3-TTS).
//!
//! Dual-track embedding (text + codec, summed position-wise), 28-layer Qwen3
//! decoder emitting codebook 0 per 12.5 Hz frame, then a 5-layer code predictor
//! fills codebooks 1..=15 autoregressively. Sampling is Gumbel-max entirely on
//! the GPU (one 16-scalar readback per frame), matching candle's sync model.

use anyhow::{Result, anyhow};
use burn::tensor::{Distribution, IndexingUpdateOp, Int, Tensor, module};

use crate::config::{CodePredictorConfig, Qwen3TTSConfig, Qwen3TTSGenerationConfig, TalkerConfig};
use crate::model::{
    DecoderLayer, Weights, causal_mask, compute_default_rope_parameters, dt, linear, rms_norm,
    rotary,
};

use burn::tensor::DType;

// ---------------------------------------------------------------------------
// Sampling (all on-device, same math as candle gpu_sample_token)
// ---------------------------------------------------------------------------

/// Sample one token from (1, vocab) logits via Gumbel-max on the GPU. Returns a
/// (1,) Int tensor on the device. `rep_mult` (1, vocab) applies the HF
/// repetition penalty (negative logits scaled up, positive scaled down).
pub fn gpu_sample_token(
    logits: Tensor<2>, // (1, vocab) f16
    do_sample: bool,
    temperature: f64,
    rep_mult: Option<&Tensor<2>>, // (1, vocab) f32
) -> Tensor<1, Int> {
    let mut lg = logits.cast(burn::tensor::DType::F32);

    if let Some(mult) = rep_mult {
        let mult = mult.clone().cast(burn::tensor::DType::F32);
        let neg = lg.clone().lower_scalar(0.0_f32);
        let penalized_neg = lg.clone() * mult.clone();
        let penalized_pos = lg.clone() / mult;
        lg = penalized_pos.mask_where(neg, penalized_neg);
    }

    if do_sample && temperature > 0.0 && (temperature - 1.0).abs() > 1e-6 {
        lg = lg / temperature;
    }

    if !do_sample {
        // Greedy: argmax (burn keeps the reduced dim; squeeze to (1,)).
        return lg.argmax(1).reshape([1]);
    }

    // Multinomial via Gumbel-max: u ~ U(1e-7, 1), g = -log(-log(u)),
    // argmax(lg + g). -inf entries stay -inf + finite → never selected.
    let u = Tensor::random_like(&lg, Distribution::Uniform(1e-7, 1.0));
    let gumbel = u.log().neg().log().neg();
    (lg + gumbel).argmax(1).reshape([1])
}

// ---------------------------------------------------------------------------
// Code predictor (5 layers, codebooks 1..=15)
// ---------------------------------------------------------------------------

pub struct CodePredictor {
    cfg: CodePredictorConfig,
    codec_embedding: Vec<Tensor<2>>, // 15 × (vocab, talker_hidden)
    layers: Vec<DecoderLayer>,       // 5 × Qwen3 (predictor hidden)
    norm_w: Tensor<1>,
    inv_freq: Vec<f32>,
    lm_head: Vec<Tensor<2>>, // 15 × (vocab, hidden)
    small_to_mtp: Option<(Tensor<2>, Tensor<1>)>,
    device: burn::tensor::Device,
}

impl CodePredictor {
    pub fn new(w: &Weights, cfg: &CodePredictorConfig, talker_hidden: usize) -> Result<Self> {
        let device = w.device();
        let mut codec_embedding = Vec::with_capacity(cfg.num_code_groups - 1);
        for g in 0..cfg.num_code_groups - 1 {
            codec_embedding.push(w.get(
                &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
                dt(),
            )?);
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                w,
                &format!("talker.code_predictor.model.layers.{i}"),
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        let norm_w = w.get("talker.code_predictor.model.norm.weight", dt())?;
        let inv_freq = compute_default_rope_parameters(cfg.head_dim, cfg.rope_theta as f32);
        let mut lm_head = Vec::with_capacity(cfg.num_code_groups - 1);
        for g in 0..cfg.num_code_groups - 1 {
            lm_head.push(w.get(&format!("talker.code_predictor.lm_head.{g}.weight"), dt())?);
        }
        // 1.7B: talker 2048 → predictor 1024; 0.6B skips the projection.
        let small_to_mtp = if talker_hidden == cfg.hidden_size {
            None
        } else {
            Some((
                w.get("talker.code_predictor.small_to_mtp_projection.weight", dt())?,
                w.get("talker.code_predictor.small_to_mtp_projection.bias", dt())?,
            ))
        };
        Ok(Self {
            cfg: cfg.clone(),
            codec_embedding,
            layers,
            norm_w,
            inv_freq,
            lm_head,
            small_to_mtp,
            device,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_cache();
        }
    }

    /// One decoder forward over `embeds` (1, T, talker_hidden) → last-position
    /// hidden (1, 1, predictor_hidden).
    fn forward_hidden(&mut self, embeds: Tensor<3>, seqlen_offset: usize) -> Tensor<3> {
        let xs = match &self.small_to_mtp {
            Some((w, b)) => linear(embeds, w.clone(), Some(b.clone())),
            None => embeds,
        };
        let [bs, seq_len, _] = xs.dims();
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(&self.device, bs, seq_len))
        };
        let (cos, sin) = rotary(&self.device, &self.inv_freq, seqlen_offset, seq_len, dt());
        let mut h = xs;
        for l in self.layers.iter_mut() {
            h = l.forward(h, &cos, &sin, mask.as_ref());
        }
        let h = rms_norm(h, self.norm_w.clone(), self.cfg.rms_norm_eps);
        h.narrow(1, seq_len - 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Talker
// ---------------------------------------------------------------------------

struct ResizeMlp {
    fc1_w: Tensor<2>,
    fc1_b: Tensor<1>,
    fc2_w: Tensor<2>,
    fc2_b: Tensor<1>,
}

impl ResizeMlp {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let h = linear(x, self.fc1_w.clone(), Some(self.fc1_b.clone()));
        let h = burn::tensor::activation::silu(h);
        linear(h, self.fc2_w.clone(), Some(self.fc2_b.clone()))
    }
}

pub struct Talker {
    cfg: TalkerConfig,
    tts: Qwen3TTSConfig,
    device: burn::tensor::Device,
    text_embedding: Tensor<2>,  // (text_vocab, text_hidden)
    text_projection: ResizeMlp, // text_hidden → hidden
    codec_embedding: Tensor<2>, // (vocab, hidden)
    layers: Vec<DecoderLayer>,  // 28 × Qwen3
    norm_w: Tensor<1>,
    inv_freq: Vec<f32>,
    codec_head: Tensor<2>, // (vocab, hidden)
    code_predictor: CodePredictor,
}

impl Talker {
    pub fn new(w: &Weights, tts: &Qwen3TTSConfig) -> Result<Self> {
        let cfg = &tts.talker_config;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                w,
                &format!("talker.model.layers.{i}"),
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        let code_predictor = CodePredictor::new(w, &cfg.code_predictor_config, cfg.hidden_size)?;
        Ok(Self {
            cfg: cfg.clone(),
            tts: tts.clone(),
            device: w.device(),
            text_embedding: w.get("talker.model.text_embedding.weight", dt())?,
            text_projection: ResizeMlp {
                fc1_w: w.get("talker.text_projection.linear_fc1.weight", dt())?,
                fc1_b: w.get("talker.text_projection.linear_fc1.bias", dt())?,
                fc2_w: w.get("talker.text_projection.linear_fc2.weight", dt())?,
                fc2_b: w.get("talker.text_projection.linear_fc2.bias", dt())?,
            },
            codec_embedding: w.get("talker.model.codec_embedding.weight", dt())?,
            layers,
            norm_w: w.get("talker.model.norm.weight", dt())?,
            inv_freq: compute_default_rope_parameters(cfg.head_dim, cfg.rope_theta as f32),
            codec_head: w.get("talker.codec_head.weight", dt())?,
            code_predictor,
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.clear_cache();
        }
    }

    // --- embedding helpers -------------------------------------------------

    /// text_embedding + text_projection for token ids → (1, N, hidden).
    fn text_embed(&self, ids: &[u32]) -> Tensor<3> {
        let t = Tensor::<1, Int>::from_ints(ids, &self.device).reshape([1, ids.len()]);
        let emb = module::embedding(self.text_embedding.clone(), t);
        self.text_projection.forward(emb)
    }

    /// codec_embedding for one id (CPU) → (1, 1, hidden).
    fn codec_embed1(&self, id: u32) -> Tensor<3> {
        let t = Tensor::<2, Int>::from_ints([[id as i32]], &self.device);
        module::embedding(self.codec_embedding.clone(), t)
    }

    /// codec_embedding for one on-device token (1,1) Int → (1, 1, hidden).
    fn codec_embed1_gpu(&self, id_t: &Tensor<2, Int>) -> Tensor<3> {
        module::embedding(self.codec_embedding.clone(), id_t.clone())
    }

    /// Sum of all 16 codebook embeddings for one frame (on-device tokens) →
    /// (1, 1, hidden). codes[0] uses the talker embedding; codes[1..16] the
    /// predictor's.
    fn frame_embed_gpu(&self, code0_t: &Tensor<2, Int>, rest_t: &[Tensor<1, Int>]) -> Tensor<3> {
        let mut acc = self.codec_embed1_gpu(code0_t);
        for (g, tok) in rest_t.iter().enumerate() {
            let t = tok.clone().reshape([1, 1]);
            let e = module::embedding(self.code_predictor.codec_embedding[g].clone(), t);
            acc = acc + e;
        }
        acc
    }

    /// One decoder forward over `embeds` (1, T, hidden) → (last hidden (1,1,hidden),
    /// codec_head logits (1, vocab)).
    fn forward_step_gpu(
        &mut self,
        embeds: Tensor<3>,
        seqlen_offset: usize,
    ) -> (Tensor<3>, Tensor<2>) {
        let [bs, seq_len, _] = embeds.dims();
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(&self.device, bs, seq_len))
        };
        let (cos, sin) = rotary(&self.device, &self.inv_freq, seqlen_offset, seq_len, dt());
        let mut h = embeds;
        for l in self.layers.iter_mut() {
            h = l.forward(h, &cos, &sin, mask.as_ref());
        }
        let h = rms_norm(h, self.norm_w.clone(), self.cfg.rms_norm_eps);
        let last = h.narrow(1, seq_len - 1, 1); // (1,1,hidden)
        let logits = linear(last.clone(), self.codec_head.clone(), None); // (1,1,vocab)
        let logits = logits.reshape([1, self.cfg.vocab_size]); // (1, vocab)
        (last, logits)
    }

    /// Generate codec frames for `text_ids` → (n_frames, 16) u32.
    pub fn generate(
        &mut self,
        text_ids: &[u32],
        language: &str,
        newline_token: u32,
        gen_cfg: &Qwen3TTSGenerationConfig,
        max_new_tokens: usize,
    ) -> Result<Tensor<2, Int>> {
        if text_ids.is_empty() {
            return Err(anyhow!("talker.generate: empty text"));
        }
        self.clear_kv_cache();
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;

        // --- special embeds (text-projected bos/eos/pad) -------------------
        let bep = self.text_embed(&[
            self.tts.tts_bos_token_id,
            self.tts.tts_eos_token_id,
            self.tts.tts_pad_token_id,
        ]); // (1,3,hidden)
        let tts_bos_embed = bep.clone().narrow(1, 0, 1);
        let tts_eos_embed = bep.clone().narrow(1, 1, 1);
        let tts_pad_embed = bep.narrow(1, 2, 1);

        // --- codec-track prefix: think/nothink block + [pad, bos] ----------
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
        let mut codec_rows: Vec<Tensor<3>> = Vec::new();
        for &id in &think_ids {
            codec_rows.push(self.codec_embed1(id));
        }
        codec_rows.push(self.codec_embed1(cfg.codec_pad_id));
        codec_rows.push(self.codec_embed1(cfg.codec_bos_id));
        let codec_prefix: Tensor<3> = Tensor::cat(codec_rows, 1); // (1, P, hidden)
        let p_len = codec_prefix.dims()[1];

        // --- role prefix <|im_start|>assistant\n + (pads+bos text) ⊕ codec prefix ---
        let role = self.text_embed(&[
            self.tts.im_start_token_id,
            self.tts.assistant_token_id,
            newline_token,
        ]); // (1,3,hidden)
        let n_pad = p_len - 2;
        let pad_rows = tts_pad_embed.clone().repeat_dim(1, n_pad); // (1, n_pad, hidden)
        let text_track_head = Tensor::cat(vec![pad_rows, tts_bos_embed], 1); // (1, P-1, hidden)
        let head = text_track_head + codec_prefix.clone().narrow(1, 0, p_len - 1); // (1, P-1, hidden)
        let prompt0 = Tensor::cat(vec![role, head], 1); // (1, 3 + (P-1), hidden)

        // --- non-ICL: first text token ⊕ last codec prefix row (bos); rest → trailing ---
        let first = self.text_embed(&text_ids[0..1]) + codec_prefix.narrow(1, p_len - 1, 1);
        let prompt = Tensor::cat(vec![prompt0, first], 1);
        let rest = if text_ids.len() > 1 {
            let body = self.text_embed(&text_ids[1..]);
            Tensor::cat(vec![body, tts_eos_embed], 1)
        } else {
            tts_eos_embed
        };
        self.generate_inner(prompt, Some(rest), tts_pad_embed, gen_cfg, max_new_tokens)
    }

    /// The AR decode loop. Returns (n_frames, 16) Int codes.
    fn generate_inner(
        &mut self,
        prompt: Tensor<3>,
        trailing: Option<Tensor<3>>,
        tts_pad_embed: Tensor<3>,
        gen_cfg: &Qwen3TTSGenerationConfig,
        max_new_tokens: usize,
    ) -> Result<Tensor<2, Int>> {
        let vocab_size = self.cfg.vocab_size;
        let codec_eos = self.cfg.codec_eos_token_id;
        let num_code_groups = self.cfg.num_code_groups;
        let mut offset = prompt.dims()[1];
        let (mut past_hidden, mut logits_gpu) = self.forward_step_gpu(prompt, 0);
        let n_trailing = match &trailing {
            Some(t) => t.dims()[1],
            None => 0,
        };

        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut gen_history: Vec<u32> = Vec::new(); // codebook-0 history for rep penalty
        let suppress_from = vocab_size - 1024; // suppress [2048, 3072) except codec_eos
        let suppress_bias: Vec<f32> = (0..vocab_size)
            .map(|i| {
                if i >= suppress_from && i != codec_eos as usize {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let suppress_bias: Tensor<2> =
            Tensor::<1>::from_floats(suppress_bias.as_slice(), &self.device)
                .reshape([1, vocab_size])
                .cast(dt());
        let rep_penalty = gen_cfg.repetition_penalty;
        let apply_rep = (rep_penalty - 1.0).abs() > 1e-6;
        let mut rep_mult: Tensor<2> =
            Tensor::<2, burn::tensor::Float>::ones([1, vocab_size], &self.device)
                .cast(burn::tensor::DType::F32);
        // QWEN3_TTS_GREEDY equivalent: BURN_TTS_GREEDY=1 forces argmax.
        let greedy = std::env::var("BURN_TTS_GREEDY").is_ok();

        for step in 0..max_new_tokens {
            // Sample codebook 0 on the GPU (suppression bias + rep penalty + temp + Gumbel).
            let lg = logits_gpu + suppress_bias.clone();
            let rep = if apply_rep && step >= 2 {
                Some(&rep_mult)
            } else {
                None
            };
            let code0_t =
                gpu_sample_token(lg, gen_cfg.do_sample && !greedy, gen_cfg.temperature, rep); // (1,) Int on device
            // Scatter the penalty into this token's multiplier column (idempotent
            // per distinct token — HF applies the penalty once per distinct token).
            if apply_rep {
                let idx = Tensor::cat(
                    vec![
                        Tensor::zeros([1, 1], &self.device),
                        code0_t.clone().reshape([1, 1]),
                    ],
                    1,
                ); // (1, 2) Int
                let pen = Tensor::from_floats([rep_penalty], &self.device); // (1,)
                rep_mult = rep_mult.scatter_nd::<2, 1>(idx, pen, IndexingUpdateOp::Assign);
            }

            // Code predictor fills codebooks 1..=15 (on-device tokens, no readback).
            let rest_t = self.code_predictor_predict(&past_hidden, &code0_t, gen_cfg)?;

            // Next input embedding: Σ16 codebook embeddings + trailing text row.
            let mut next = self.frame_embed_gpu(&code0_t.clone().reshape([1, 1]), &rest_t);
            let text_row = match &trailing {
                Some(tr) if step < n_trailing => tr.clone().narrow(1, step, 1),
                _ => tts_pad_embed.clone(),
            };
            next = next + text_row;

            // Single per-frame readback: cat 16 tokens, read once.
            let mut all_t = vec![code0_t.clone().reshape([1])];
            all_t.extend(rest_t.iter().map(|t| t.clone().reshape([1])));
            let flat: Tensor<1, Int> = Tensor::cat(all_t, 0);
            let flat_vec: Vec<i32> = flat
                .to_data()
                .to_vec::<i32>()
                .map_err(|e| anyhow::anyhow!("frame readback: {e}"))?;
            let flat_vec: Vec<u32> = flat_vec.iter().map(|&v| v as u32).collect();
            let code0 = flat_vec[0];

            // EOS check AFTER the predictor (matches mlx/candle).
            if code0 == codec_eos && step >= 2 {
                break;
            }
            gen_history.push(code0);
            frames.push(flat_vec);

            let (h, l) = self.forward_step_gpu(next, offset);
            past_hidden = h;
            logits_gpu = l;
            offset += 1;
        }

        let flat: Vec<i32> = frames.iter().flatten().map(|&v| v as i32).collect();
        let n = flat.len() / num_code_groups;
        let t = Tensor::<1, Int>::from_ints(flat.as_slice(), &self.device)
            .reshape([n, num_code_groups]);
        Ok(t)
    }

    /// Run the code predictor for one frame (codebook 0 known) → 15 on-device
    /// (1,) Int tokens, strictly sequential (each step's token feeds the next).
    fn code_predictor_predict(
        &mut self,
        talker_hidden_last: &Tensor<3>,
        code0_t: &Tensor<1, Int>,
        gen_cfg: &Qwen3TTSGenerationConfig,
    ) -> Result<Vec<Tensor<1, Int>>> {
        let code0_emb = self.codec_embed1_gpu(&code0_t.clone().reshape([1, 1]));
        let cp = &mut self.code_predictor;
        cp.clear_kv_cache();
        let prefill = Tensor::cat(vec![talker_hidden_last.clone(), code0_emb], 1); // (1,2,hidden)
        let mut hidden = cp.forward_hidden(prefill, 0); // (1,1,predictor_hidden) from position 1
        let n_groups = cp.cfg.num_code_groups - 1;
        let mut offset = 2usize;
        let mut tokens: Vec<Tensor<1, Int>> = Vec::with_capacity(n_groups);
        for g in 0..n_groups {
            if std::env::var("BURN_TTS_DEBUG").is_ok() && g == 2 {
                let v: Vec<f32> = hidden
                    .clone()
                    .cast(DType::F32)
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap();
                eprintln!("[debug cp g=2 hidden] {:?}", &v[..8.min(v.len())]);
            }
            let logits =
                linear(hidden.clone(), cp.lm_head[g].clone(), None).reshape([1, cp.cfg.vocab_size]); // (1, vocab)
            if std::env::var("BURN_TTS_DEBUG").is_ok() && g == 3 {
                let v: Vec<f32> = logits
                    .clone()
                    .cast(DType::F32)
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap();
                let mut top: Vec<(f32, u32)> =
                    v.iter().enumerate().map(|(i, &x)| (x, i as u32)).collect();
                top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                eprintln!("[debug cp g=3 logits top5] {:?}", &top[..5]);
            }
            let token = gpu_sample_token(
                logits,
                gen_cfg.subtalker_dosample && std::env::var("BURN_TTS_GREEDY").is_err(),
                gen_cfg.subtalker_temperature,
                None, // code predictor applies no repetition penalty
            ); // (1,) Int on device
            if g + 1 < n_groups {
                let emb =
                    module::embedding(cp.codec_embedding[g].clone(), token.clone().reshape([1, 1]));
                hidden = cp.forward_hidden(emb, offset);
                offset += 1;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}
