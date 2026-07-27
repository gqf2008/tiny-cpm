//! Ported from aha (github.com/jhqxxx/aha) src/models/moss_tts_nano/model.rs
//! Adaptations vs aha:
//! - `generate()` takes `sample_len` (max codec frames) and `save_path` parameters
//!   instead of the hardcoded `sample_len = 100` and `./demo.wav`; it returns the
//!   number of frames actually generated. The frame loop + codec decode also live
//!   in `generate_waveform()`, which returns the waveform in memory (used by the
//!   `live` subcommand); `generate()` is a thin wrapper that adds `save_wav`.
use crate::{
    common::sample::{sample_from_logits_vec, simple_sample_cpu},
    models::{
        gpt2::GPT2Model, moss_audio_tokenizer_nano::MossAudioTokenizer,
        moss_tts_nano::config::MossTTSConfig,
    },
    utils::audio_utils::save_wav,
};
use anyhow::{Result, anyhow};
use candle_core::{D, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear_no_bias};
// use candle_transformers::generation::LogitsProcessor;

#[derive(PartialEq, Clone, Debug)]
pub enum MossTTSMode {
    Continuation,
    VoiceClone,
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Timing stats from `MossTTSModel::generate`.
pub struct MossGenStats {
    /// Codec frames actually generated (stops early on the end token).
    pub frames: usize,
    /// Time to first frame: prompt prefill + first full codec frame.
    pub ttft: std::time::Duration,
    /// Codec (audio tokenizer) decode of all frames to waveform.
    pub codec_decode: std::time::Duration,
    /// Whole generate() call.
    pub total: std::time::Duration,
}

/// Streaming sink for `MossTTSModel::generate_waveform`: every `chunk_frames`
/// codec frames, the prefix generated so far is decoded and the new audio
/// tail is handed to `on_chunk` (interleaved stereo f32, [-1,1]). `on_chunk`
/// returns `false` to request an early abort (used by `live --barge-in` to
/// cut off synthesis mid-utterance).
pub struct StreamChunk<'a> {
    pub chunk_frames: usize,
    pub on_chunk: &'a mut dyn FnMut(Vec<f32>) -> bool,
}

/// Emit the not-yet-emitted tail of a (channels, len) waveform as interleaved
/// f32. Returns `(emitted, continue)` — `continue` is `false` if `on_chunk`
/// asked to abort.
fn emit_tail(
    waveform: &Tensor,
    emitted: usize,
    on_chunk: &mut dyn FnMut(Vec<f32>) -> bool,
) -> Result<(usize, bool)> {
    let (channels, len) = waveform.dims2()?;
    if len <= emitted {
        return Ok((emitted, true));
    }
    // One bulk GPU→CPU copy, then interleave on the host.
    let w = waveform
        .to_dtype(candle_core::DType::F32)?
        .to_vec2::<f32>()?;
    let mut pcm = Vec::with_capacity((len - emitted) * channels);
    for i in emitted..len {
        for row in w.iter().take(channels) {
            pcm.push(row[i]);
        }
    }
    let cont = on_chunk(pcm);
    Ok((len, cont))
}

pub struct MossTTSModel {
    transformer: GPT2Model,
    audio_embeddings: Vec<Embedding>,
    text_lm_head: Linear,
    audio_lm_heads: Vec<Linear>,
    local_transformer: GPT2Model,
    audio_assistant_slot_token_id: usize,
    audio_end_token_id: usize,
    n_vq: usize,
    audio_pad_token_id_tensor: Tensor,
    audio_codebook_sizes: Vec<usize>,
    audio_temperature: f64,
    audio_top_k: usize,
    audio_top_p: f32,
    audio_repetition_penalty: f32,
    // audio_processor: LogitsProcessor,
}

impl MossTTSModel {
    pub fn new(vb: VarBuilder, cfg: &MossTTSConfig) -> Result<Self> {
        let transformer = GPT2Model::new(
            vb.pp("transformer"),
            cfg.gpt2_config.n_embd,
            cfg.gpt2_config.n_head,
            cfg.gpt2_config.n_layer,
            cfg.gpt2_config.vocab_size,
            // cfg.gpt2_config.n_positions,
        )?;
        let mut audio_embeddings = vec![];
        let audio_embed_vb = vb.pp("audio_embeddings");
        for i in 0..cfg.n_vq {
            let embed = embedding(
                cfg.audio_codebook_sizes[i],
                cfg.gpt2_config.n_embd,
                audio_embed_vb.pp(i),
            )?;
            audio_embeddings.push(embed);
        }
        let text_lm_head = linear_no_bias(
            cfg.gpt2_config.n_embd,
            cfg.gpt2_config.vocab_size,
            vb.pp("text_lm_head"),
        )?;

        let mut audio_lm_heads = vec![];
        let audio_lm_vb = vb.pp("audio_lm_heads");
        for i in 0..cfg.n_vq {
            let layer = linear_no_bias(
                cfg.gpt2_config.n_embd,
                cfg.audio_codebook_sizes[i],
                audio_lm_vb.pp(i),
            )?;
            audio_lm_heads.push(layer);
        }

        let mut local_gpt2_cfg = cfg.gpt2_config.clone();
        local_gpt2_cfg.n_layer = cfg.local_transformer_layers;
        local_gpt2_cfg.n_positions = cfg.n_vq + 1;
        local_gpt2_cfg.n_ctx = cfg.n_vq + 1;
        let local_transformer = GPT2Model::new_without_wte(
            vb.pp("local_transformer"),
            local_gpt2_cfg.n_embd,
            local_gpt2_cfg.n_head,
            local_gpt2_cfg.n_layer,
            local_gpt2_cfg.vocab_size,
            // local_gpt2_cfg.n_positions,
        )?;
        let audio_pad_token_id_tensor = Tensor::new(cfg.audio_pad_token_id, vb.device())?;
        // let audio_processor = get_logit_processor(Some(0.8), Some(0.95), Some(25), 34562);
        Ok(Self {
            transformer,
            audio_embeddings,
            text_lm_head,
            audio_lm_heads,
            local_transformer,
            audio_assistant_slot_token_id: cfg.audio_assistant_slot_token_id as usize,
            audio_end_token_id: cfg.audio_end_token_id as usize,
            n_vq: cfg.n_vq,
            audio_pad_token_id_tensor,
            audio_codebook_sizes: cfg.audio_codebook_sizes.clone(),
            // Sampling defaults: the official Python library generate() defaults
            // (1.7/0.8/25/1.0, same as audio.cpp) — A/B listening tests preferred
            // them over the official infer.py CLI set (0.8/0.95/25/1.2 via aha).
            // Override via MOSS_TEMPERATURE / MOSS_TOP_P / MOSS_TOP_K /
            // MOSS_REP_PENALTY.
            audio_temperature: env_f64("MOSS_TEMPERATURE", 1.7),
            audio_top_k: env_usize("MOSS_TOP_K", 25),
            audio_top_p: env_f32("MOSS_TOP_P", 0.8),
            audio_repetition_penalty: env_f32("MOSS_REP_PENALTY", 1.0),
            // audio_processor,
        })
    }

    fn build_inputs_embeds(&self, input_ids: &Tensor) -> Result<Tensor> {
        let text_ids = input_ids.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let mut inputs_embeds = if let Some(wte) = &self.transformer.wte {
            wte.forward(&text_ids)?
        } else {
            return Err(anyhow!("MossTTS transformer wte can not be none"));
        };
        for (channel_index, embedding) in self.audio_embeddings.iter().enumerate() {
            let channel_ids = input_ids
                .narrow(D::Minus1, channel_index + 1, 1)?
                .squeeze(D::Minus1)?;
            let valid_mask = channel_ids.ne(&self
                .audio_pad_token_id_tensor
                .broadcast_as(channel_ids.shape())?)?;
            let invalid_mask = channel_ids.lt(&channel_ids.zeros_like()?)?;
            let embedding_nums = Tensor::new(
                self.audio_codebook_sizes[channel_index] as u32,
                input_ids.device(),
            )?;
            let invalid_mask1 =
                channel_ids.ge(&embedding_nums.broadcast_as(channel_ids.shape())?)?;
            let invalid_mask = valid_mask
                .minimum(&invalid_mask.maximum(&invalid_mask1)?)?
                .to_dtype(candle_core::DType::U32)?;
            if invalid_mask.sum_all()?.to_scalar::<u32>()? > 0 {
                return Err(anyhow!("Found out-of-range audio token ids for channel"));
            }
            let safe_ids = valid_mask.where_cond(&channel_ids, &channel_ids.zeros_like()?)?;
            let audio_embeds = embedding.forward(&safe_ids)?;
            let audio_embeds = audio_embeds.broadcast_mul(
                &valid_mask
                    .unsqueeze(D::Minus1)?
                    .to_dtype(audio_embeds.dtype())?,
            )?;
            inputs_embeds = inputs_embeds.add(&audio_embeds)?;
        }
        Ok(inputs_embeds)
    }

    fn sample_next_assistant_text_token(&self, logits: &Tensor) -> Result<usize> {
        let logits = logits.squeeze(0)?.squeeze(0)?;
        // One sync for the two candidate logits, then sample on CPU.
        let idx = Tensor::new(
            &[
                self.audio_assistant_slot_token_id as u32,
                self.audio_end_token_id as u32,
            ],
            logits.device(),
        )?;
        let pair = logits
            .index_select(&idx, 0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        let token = sample_from_logits_vec(&pair, true, None, None, None, None, 1.0)?;
        if token == 0 {
            Ok(self.audio_assistant_slot_token_id)
        } else {
            Ok(self.audio_end_token_id)
        }
    }

    fn build_generation_row(&self, audio_token_ids: &Tensor) -> Result<Tensor> {
        let slot = Tensor::from_slice(
            &[self.audio_assistant_slot_token_id as u32],
            (1, 1, 1),
            audio_token_ids.device(),
        )?;
        let audio_token_ids = audio_token_ids.unsqueeze(0)?.unsqueeze(0)?;
        Ok(Tensor::cat(&[&slot, &audio_token_ids], D::Minus1)?)
    }

    /// Generates up to `sample_len` codec frames and decodes them with the
    /// audio tokenizer. Returns the (channels, len) f32 waveform (not written
    /// to disk) plus generation stats (frame count, TTFT, codec decode time).
    ///
    /// When `stream` is set, every `stream.chunk_frames` frames the frames
    /// generated so far are decoded through the (causal) codec and the newly
    /// available interleaved-stereo f32 tail is passed to `stream.on_chunk` —
    /// this lets playback start after the first chunk instead of after the
    /// whole utterance. Prefix re-decode is exact because the codec is causal;
    /// it costs one extra codec pass per chunk.
    pub fn generate_waveform(
        &mut self,
        input_ids: &Tensor,
        audio_tokenizer: &MossAudioTokenizer,
        sample_len: usize,
        mut stream: Option<StreamChunk<'_>>,
    ) -> Result<(Tensor, MossGenStats)> {
        let gen_start = std::time::Instant::now();
        // generate() restarts seqlen_offset at 0, so any KV cache left over
        // from a previous call would corrupt attention (mask/kv length
        // mismatch). Clear both transformers' caches up front — required for
        // calling synthesize() more than once on the same model instance.
        self.transformer.clear_kv_cache();
        self.local_transformer.clear_kv_cache();
        let mut ttft = None;
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let mut generated_frames = vec![];
        // Samples (per channel) already emitted through `stream`.
        let mut emitted = 0usize;
        // Python reference (modeling_moss_tts_nano.py:1687, 1733-1735): the
        // repetition penalty for channel k is applied against that channel's
        // previously generated frames only (`generated_audio_history[:, :, k]`;
        // frames generated in this call, prompt frames are NOT included).
        let mut audio_history: Vec<Vec<u32>> = vec![Vec::new(); self.n_vq];
        let mut current_model_input_ids = input_ids.clone();
        let mut aborted = false;
        for _ in 0..sample_len {
            let inputs_embeds = self.build_inputs_embeds(&current_model_input_ids)?;
            let outputs = self.transformer.forward(&inputs_embeds, seqlen_offset)?;
            let outputs_len = outputs.dim(1)?;
            let global_hidden_state = outputs.narrow(1, outputs_len - 1, 1)?;
            let mut local_positions = 0usize;
            let local_outputs = self
                .local_transformer
                .forward(&global_hidden_state, local_positions)?;
            let local_len = local_outputs.dim(1)?;
            let local_hidden_states = local_outputs.narrow(1, local_len - 1, 1)?;
            let text_logits = self.text_lm_head.forward(&local_hidden_states)?;
            let next_text_token = self.sample_next_assistant_text_token(&text_logits)?;
            if next_text_token == self.audio_end_token_id {
                self.local_transformer.clear_kv_cache();
                break;
            }
            let mut next_frame_tokens = vec![];
            let mut current_local_input = if let Some(wte) = &self.transformer.wte {
                wte.forward(&Tensor::from_slice(
                    &[next_text_token as u32],
                    (1, 1),
                    input_ids.device(),
                )?)?
            } else {
                return Err(anyhow!("MossTTS GPT2 wte can not be none"));
            };
            for channel_index in 0..self.n_vq {
                local_positions += 1;
                let local_outputs = self
                    .local_transformer
                    .forward(&current_local_input, local_positions)?;
                let local_len = local_outputs.dim(1)?;
                let local_hidden_states = local_outputs.narrow(1, local_len - 1, 1)?;
                let channel_logits = self.audio_lm_heads[channel_index]
                    .forward(&local_hidden_states)?
                    .squeeze(0)?
                    .squeeze(0)?;
                // One GPU→CPU sync for the logits, then top-k/top-p/penalty
                // and multinomial sampling all on CPU (small codebook — this
                // avoids ~15-20 tiny GPU kernels + syncs per channel).
                let channel_token = simple_sample_cpu(
                    &channel_logits,
                    true,
                    Some(self.audio_temperature),
                    Some(self.audio_top_k),
                    Some(self.audio_top_p),
                    Some(&audio_history[channel_index]),
                    self.audio_repetition_penalty,
                )?;
                next_frame_tokens.push(channel_token);
                audio_history[channel_index].push(channel_token);
                current_local_input = self.audio_embeddings[channel_index].forward(
                    &Tensor::from_slice(&[channel_token], (1, 1), input_ids.device())?,
                )?;
            }
            self.local_transformer.clear_kv_cache();
            let next_frame = Tensor::new(next_frame_tokens, input_ids.device())?;
            current_model_input_ids = self.build_generation_row(&next_frame)?;
            seqlen_offset += seq_len;
            seq_len = 1;
            generated_frames.push(next_frame);
            if generated_frames.len() == 1 {
                // TTFT: prompt prefill + first full codec frame.
                ttft = Some(gen_start.elapsed());
            }
            if let Some(s) = stream.as_mut()
                && generated_frames.len() % s.chunk_frames == 0
            {
                let audio_token_ids = Tensor::stack(&generated_frames, 0)?;
                let waveform = audio_tokenizer
                    .decode_audio_token_ids_to_waveform(&audio_token_ids)?
                    .squeeze(0)?;
                let (new_emitted, cont) = emit_tail(&waveform, emitted, &mut s.on_chunk)?;
                emitted = new_emitted;
                if !cont {
                    aborted = true;
                    break;
                }
            }
        }
        let num_frames = generated_frames.len();
        if num_frames == 0 {
            // The model emitted audio_end on the very first frame (happens on
            // empty/degenerate input) — Tensor::stack would panic below.
            return Err(anyhow!(
                "MOSS generated 0 codec frames (immediate end token); nothing to decode"
            ));
        }
        let decode_start = std::time::Instant::now();
        let audio_token_ids = Tensor::stack(&generated_frames, 0)?;
        let waveform = audio_tokenizer
            .decode_audio_token_ids_to_waveform(&audio_token_ids)?
            .squeeze(0)?;
        if let Some(s) = stream.as_mut() {
            if !aborted {
                emit_tail(&waveform, emitted, &mut s.on_chunk)?;
            }
        }
        let stats = MossGenStats {
            frames: num_frames,
            ttft: ttft.unwrap_or_default(),
            codec_decode: decode_start.elapsed(),
            total: gen_start.elapsed(),
        };
        Ok((waveform, stats))
    }

    /// `generate_waveform` + write the (stereo) waveform to `save_path`.
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        audio_tokenizer: &MossAudioTokenizer,
        sample_len: usize,
        save_path: &str,
    ) -> Result<MossGenStats> {
        let (waveform, stats) =
            self.generate_waveform(input_ids, audio_tokenizer, sample_len, None)?;
        save_wav(
            &waveform,
            save_path,
            2,
            audio_tokenizer.sampling_rate as u32,
        )?;
        Ok(stats)
    }

    #[allow(dead_code)]
    pub fn decode(
        &self,
        prompt_audio_code: Option<&Tensor>,
        audio_tokenizer: &MossAudioTokenizer,
    ) -> Result<()> {
        if let Some(audio) = prompt_audio_code {
            let waveform = audio_tokenizer
                .decode_audio_token_ids_to_waveform(audio)?
                .squeeze(0)?;
            save_wav(
                &waveform,
                "./demo.wav",
                2,
                audio_tokenizer.sampling_rate as u32,
            )?;
        }
        Ok(())
    }
}
