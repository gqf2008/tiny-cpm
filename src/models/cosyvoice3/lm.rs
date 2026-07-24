//! CosyVoice3 lm (see mod.rs header).
//!
//! CosyVoice3LM: Qwen2-0.5B backbone + speech-token embedding/head, ported
//! from CrispASR `src/cosyvoice3_tts.cpp` (hparams `cv3_hp`, embedding
//! assembly `cv3_build_lm_input_embeds`, AR loop
//! `cv3_generate_tokens_with_stop_floor`, RAS sampler
//! `cosyvoice3_tts_sample_ras` — itself a port of upstream CosyVoice
//! `cosyvoice/utils/common.py::ras_sampling`).
//!
//! Weights load from the ORIGINAL `llm.pt` pickle with upstream tensor names
//! (`llm.model.model.*`, `llm.model.lm_head.weight`, `speech_embedding.weight`,
//! `llm_decoder.weight`); the `llm.model.` prefix is stripped so the backbone
//! names line up with the `crate::models::qwen2` module tree. Runs F32 (the
//! Python reference also runs `.float()`).
//!
//! When `cosyvoice3-llm-q4_k.gguf` exists in the model dir it takes
//! precedence: the Qwen2 backbone runs QMatMul (see `quantized_lm.rs`), with
//! F32 activations as on the unquantized path, so generation semantics
//! (RAS sampler, KV-cached AR loop) are unchanged.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor, pickle::read_all_with_key};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear_no_bias};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::models::qwen2::{Qwen2Config, Qwen2Decoder};
use crate::utils::tensor_utils::prepare_causal_attention_mask;

use super::quantized_lm;

/// Backbone dispatch: verbatim aha Qwen2 port (F32, from llm.pt) or the
/// QMatMul mirror (GGUF Q4_K, same forward semantics).
enum Qwen2Backbone {
    Full(Qwen2Decoder),
    Quant(quantized_lm::QuantizedQwen2Decoder),
}

impl Qwen2Backbone {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Full(d) => d.forward(xs, attention_mask, seqlen_offset),
            Self::Quant(d) => d.forward(xs, attention_mask, seqlen_offset),
        }
    }
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Full(d) => d.clear_kv_cache(),
            Self::Quant(d) => d.clear_kv_cache(),
        }
    }
}

/// Speech codebook size. Ids in [SPEECH_CODEBOOK, speech_vocab) are specials
/// (sos/eos/task_id/fill/...); sampling any of them ends decoding.
pub const SPEECH_CODEBOOK: usize = 6561;
/// sos / task_id markers live in the SPEECH embedding table (upstream
/// CosyVoice3LM: sos = speech_token_size + 0, task_id = + 2).
const SOS_ID: u32 = SPEECH_CODEBOOK as u32; // 6561
const TASK_ID: u32 = SPEECH_CODEBOOK as u32 + 2; // 6563

// RAS defaults (upstream ras_sampling / C++ cv3 default params).
const RAS_TOP_P: f32 = 0.8;
const RAS_TOP_K: usize = 25;
const RAS_WIN_SIZE: usize = 10;
const RAS_TAU_R: f32 = 0.1;

pub struct CosyVoice3LM {
    cfg: Qwen2Config,
    token_embedding: Embedding,
    decoder: Qwen2Backbone,
    speech_embedding: Embedding,
    speech_lm_head: Linear,
    speech_vocab: usize,
    device: Device,
}

impl CosyVoice3LM {
    /// `dir` = Fun-CosyVoice3-0.5B-2512 model dir (holds `llm.pt` and
    /// `CosyVoice-BlankEN/config.json`; hparams fall back to the reference
    /// `cv3_hp` defaults when the config is absent).
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg_path = dir.join("CosyVoice-BlankEN").join("config.json");
        let cfg = match std::fs::read_to_string(&cfg_path) {
            Ok(s) => serde_json::from_str::<Qwen2Config>(&s)
                .with_context(|| format!("parse {}", cfg_path.display()))?,
            Err(_) => default_qwen2_config(),
        };

        // Prefer the quantized GGUF (QMatMul Q4_K) when present; fall back to
        // the original F32 llm.pt pickle otherwise.
        let gguf_path = dir.join("cosyvoice3-llm-q4_k.gguf");
        if gguf_path.exists() {
            let w = quantized_lm::load_gguf(&gguf_path, &cfg, device)?;
            return Ok(Self {
                cfg,
                token_embedding: w.token_embedding,
                decoder: Qwen2Backbone::Quant(w.decoder),
                speech_embedding: w.speech_embedding,
                speech_lm_head: w.speech_lm_head,
                speech_vocab: w.speech_vocab,
                device: device.clone(),
            });
        }

        let pt_path = dir.join("llm.pt");
        let named = match read_all_with_key(&pt_path, Some("state_dict")) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("llm.pt read_all_with_key(state_dict): {e}, retry with None");
                read_all_with_key(&pt_path, None)
                    .with_context(|| format!("load {}", pt_path.display()))?
            }
        };
        // Strip "llm.model." -> backbone addresses match the qwen2 module
        // tree ("model.embed_tokens", "model.layers.N", "model.norm",
        // "lm_head"); the two speech-side tables keep their names.
        let tensors: HashMap<String, Tensor> = named
            .into_iter()
            .map(|(k, t)| {
                let k = k.strip_prefix("llm.model.").map(str::to_owned).unwrap_or(k);
                (k, t)
            })
            .collect();
        let (speech_vocab, speech_dim) = tensors
            .get("llm_decoder.weight")
            .ok_or_else(|| anyhow!("llm.pt missing llm_decoder.weight"))?
            .dims2()
            .context("llm_decoder.weight dims")?;
        if speech_dim != cfg.hidden_size {
            return Err(anyhow!(
                "llm_decoder.weight in_dim {speech_dim} != hidden_size {}",
                cfg.hidden_size
            ));
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        let token_embedding =
            embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let decoder = Qwen2Decoder::new(vb.pp("model"), &cfg)?;
        let speech_embedding = embedding(speech_vocab, cfg.hidden_size, vb.pp("speech_embedding"))?;
        let speech_lm_head = linear_no_bias(cfg.hidden_size, speech_vocab, vb.pp("llm_decoder"))?;
        Ok(Self {
            cfg,
            token_embedding,
            decoder: Qwen2Backbone::Full(decoder),
            speech_embedding,
            speech_lm_head,
            speech_vocab,
            device: device.clone(),
        })
    }

    pub fn speech_vocab(&self) -> usize {
        self.speech_vocab
    }

    fn speech_embed_rows(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_slice(ids, (ids.len(),), &self.device)?;
        Ok(self.speech_embedding.forward(&ids)?)
    }

    /// One decoder pass over `inputs_embeds` (1, T, d) at `seqlen_offset`;
    /// returns the last-position speech logits as a host Vec (speech_vocab).
    fn forward_logits(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Vec<f32>> {
        let seq_len = inputs_embeds.dim(1)?;
        // T=1 steps attend to the whole cache; only prefill needs a mask.
        let mask = if seq_len == 1 {
            None
        } else {
            Some(prepare_causal_attention_mask(
                1,
                seq_len,
                seqlen_offset,
                &self.device,
            )?)
        };
        let hidden = self
            .decoder
            .forward(inputs_embeds, mask.as_ref(), seqlen_offset)?;
        // The AR head only needs the last position (C++ slices at T-1).
        let last = hidden.narrow(1, seq_len - 1, 1)?;
        let logits = self.speech_lm_head.forward(&last)?;
        Ok(logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// AR-generate speech tokens for `text_ids` (ready-made Qwen2 BPE ids,
    /// prompt text included) conditioned on `prompt_speech_tokens`.
    /// `max_steps == 0` -> upstream default 20 * n_text_tokens (floored at
    /// 16). Returns only the generated speech tokens: no sos/task/prompt,
    /// and without the terminal stop id.
    ///
    /// The RNG is re-seeded from `seed` on every call (deterministic per
    /// text), unlike CrispASR's persistent stream; the flow's Euler x_init
    /// draws from the same per-call seed — deliberate for a single-shot CLI.
    pub fn generate_speech_tokens(
        &mut self,
        text_ids: &[u32],
        prompt_speech_tokens: &[u32],
        max_steps: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        self.generate_speech_tokens_streaming(
            text_ids,
            prompt_speech_tokens,
            max_steps,
            seed,
            &mut |_| Ok(()),
        )
    }

    /// Streaming variant of `generate_speech_tokens`: identical RAS/stop
    /// semantics, but `on_token` fires after each generated token is pushed
    /// (so the callback observes the tokens in order and can flush audio
    /// chunks mid-generation, upstream cosyvoice/cli/model.py:346-361).
    pub fn generate_speech_tokens_streaming(
        &mut self,
        text_ids: &[u32],
        prompt_speech_tokens: &[u32],
        max_steps: usize,
        seed: u64,
        on_token: &mut dyn FnMut(u32) -> Result<()>,
    ) -> Result<Vec<u32>> {
        if text_ids.is_empty() {
            return Err(anyhow!("generate_speech_tokens: text_ids is empty"));
        }
        for &t in text_ids {
            if t as usize >= self.cfg.vocab_size {
                return Err(anyhow!("text id {t} out of vocab {}", self.cfg.vocab_size));
            }
        }
        for &t in prompt_speech_tokens {
            if t as usize >= SPEECH_CODEBOOK {
                return Err(anyhow!(
                    "prompt speech token {t} out of codebook {SPEECH_CODEBOOK}"
                ));
            }
        }
        let max_steps = if max_steps == 0 {
            (20 * text_ids.len()).max(16)
        } else {
            max_steps
        };
        // C++ reference: seed 0 means "use the default".
        let seed = if seed == 0 { 42 } else { seed };
        let mut rng = StdRng::seed_from_u64(seed);

        // inputs_embeds = [speech_embd[sos] | token_embd[text] |
        //                  speech_embd[task_id] | speech_embd[prompt_speech]]
        let text = Tensor::from_slice(text_ids, (text_ids.len(),), &self.device)?;
        let mut parts = vec![
            self.speech_embed_rows(&[SOS_ID])?,
            self.token_embedding.forward(&text)?,
            self.speech_embed_rows(&[TASK_ID])?,
        ];
        if !prompt_speech_tokens.is_empty() {
            parts.push(self.speech_embed_rows(prompt_speech_tokens)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let inputs_embeds = Tensor::cat(&refs, 0)?.unsqueeze(0)?; // (1, T, d)

        self.decoder.clear_kv_cache();
        let mut n_past = inputs_embeds.dim(1)?;
        let mut logits = self.forward_logits(&inputs_embeds, 0)?;

        let mut out: Vec<u32> = Vec::new();
        for _ in 0..max_steps {
            let pick = ras_sample(&logits, &out, &mut rng)?;
            // Stop floor: any id >= SPEECH_CODEBOOK is a special/stop marker.
            if pick as usize >= SPEECH_CODEBOOK {
                break;
            }
            out.push(pick);
            on_token(pick)?;
            let emb = self.speech_embed_rows(&[pick])?.unsqueeze(0)?;
            logits = self.forward_logits(&emb, n_past)?;
            n_past += 1;
        }
        Ok(out)
    }
}

/// cv3_hp fallback (vanilla Qwen2-0.5B as configured for CosyVoice3).
fn default_qwen2_config() -> Qwen2Config {
    Qwen2Config {
        vocab_size: 151936,
        hidden_size: 896,
        intermediate_size: 4864,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        max_position_embeddings: 32768,
        sliding_window: 32768,
        max_window_layers: 24,
        tie_word_embeddings: true,
        rope_theta: 1e6,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    }
}

// ---------------------------------------------------------------------------
// RAS — Repetition-Aware Sampling (VALL-E 2), port of upstream
// cosyvoice/utils/common.py::{nucleus_sampling, ras_sampling} via the C++
// reference. No temperature scaling: upstream applies softmax directly to
// the (log-prob) scores; greedy is not supported (broken upstream).
// ---------------------------------------------------------------------------

/// Stable softmax in place (subtract max first).
fn softmax_inplace(v: &mut [f32]) {
    let vmax = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f64;
    for x in v.iter_mut() {
        *x = (*x - vmax).exp();
        s += *x as f64;
    }
    if s > 0.0 {
        let inv = (1.0 / s) as f32;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Multinomial draw over unnormalised weights (torch.multinomial semantics:
/// weights / sum as the categorical distribution, inverse-CDF draw).
fn multinomial_pick(weights: &[f32], rng: &mut StdRng) -> Option<usize> {
    let sum: f64 = weights
        .iter()
        .filter(|w| w.is_finite() && **w > 0.0)
        .map(|w| *w as f64)
        .sum();
    if !(sum > 0.0) {
        return None;
    }
    let r = rng.random::<f64>() * sum;
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        if !(w.is_finite() && *w > 0.0) {
            continue;
        }
        acc += *w as f64;
        if r <= acc {
            return Some(i);
        }
    }
    Some(weights.len() - 1) // floating-point tail
}

/// Upstream `nucleus_sampling`: stable-sort probs descending, keep while
/// cum_prob < top_p AND count < top_k, multinomial over the kept
/// UNRENORMALISED probs.
fn nucleus_sample(logits: &[f32], top_p: f32, top_k: usize, rng: &mut StdRng) -> Option<u32> {
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    // Stable sort: ties keep the original index order (torch sort stable=True).
    let mut idx: Vec<u32> = (0..probs.len() as u32).collect();
    idx.sort_by(|a, b| probs[*b as usize].total_cmp(&probs[*a as usize]));
    let mut kept: Vec<f32> = Vec::new();
    let mut kept_ids: Vec<u32> = Vec::new();
    let mut cum = 0.0f64;
    for &i in &idx {
        if cum >= top_p as f64 || kept.len() >= top_k {
            break;
        }
        let p = probs[i as usize];
        cum += p as f64;
        kept.push(p);
        kept_ids.push(i);
    }
    let pick = multinomial_pick(&kept, rng)?;
    Some(kept_ids[pick])
}

/// Upstream `ras_sampling`: nucleus sample, then if the pick appears
/// >= win_size * tau_r times in the trailing win_size of the history,
/// suppress it (logit = -inf) and re-sample plain softmax-multinomial over
/// the FULL distribution.
fn ras_sample(logits: &[f32], history: &[u32], rng: &mut StdRng) -> Result<u32> {
    let mut pick = nucleus_sample(logits, RAS_TOP_P, RAS_TOP_K, rng)
        .ok_or_else(|| anyhow!("ras nucleus sample failed"))?;
    let start = history.len().saturating_sub(RAS_WIN_SIZE);
    let rep = history[start..].iter().filter(|&&t| t == pick).count();
    if rep as f32 >= RAS_WIN_SIZE as f32 * RAS_TAU_R {
        let mut modified = logits.to_vec();
        modified[pick as usize] = f32::NEG_INFINITY;
        softmax_inplace(&mut modified);
        pick = multinomial_pick(&modified, rng)
            .map(|i| i as u32)
            .ok_or_else(|| anyhow!("ras resample failed"))?;
    }
    Ok(pick)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_cv3_hp() {
        let cfg = default_qwen2_config();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.intermediate_size, 4864);
        assert_eq!(cfg.hidden_size / cfg.num_attention_heads, 64); // head_dim
        assert!((cfg.rope_theta - 1e6).abs() < 1.0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, -1.0, 0.5];
        softmax_inplace(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nucleus_respects_top_k() {
        // 100 tokens, uniform-ish logits; top_k=25 caps the kept set so the
        // multinomial can never land past the first 25 sorted entries.
        let logits: Vec<f32> = (0..100).map(|i| -(i as f32) * 0.01).collect();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..64 {
            let pick = nucleus_sample(&logits, 1.0, 25, &mut rng).unwrap();
            assert!((pick as usize) < 25);
        }
    }

    #[test]
    fn ras_suppresses_repeated_pick() {
        // Token 3 dominates the logits and already fills the repetition
        // window: RAS must suppress it and pick something else.
        let mut logits = vec![0.0f32; 16];
        logits[3] = 10.0;
        logits[5] = 5.0;
        let history = vec![3u32; RAS_WIN_SIZE];
        let mut rng = StdRng::seed_from_u64(1);
        let pick = ras_sample(&logits, &history, &mut rng).unwrap();
        assert_ne!(pick, 3);
        assert_eq!(pick, 5);
    }
}
