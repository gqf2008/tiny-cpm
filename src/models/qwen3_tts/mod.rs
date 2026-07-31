//! Qwen3-TTS 12Hz 1.7B Base, ported from github.com/QwenLM/Qwen3-TTS
//! (the `qwen_tts` package — the only reference implementation; HuggingFace
//! `transformers` has no `qwen3_tts` module).
//!
//! Components:
//! - `talker` — the LM: a stock Qwen3 decoder (RMSNorm/RoPE/GQA/QK-norm/SwiGLU)
//!   with a fused text+codec embedding and a `codec_head`; emits codec codebook 0
//!   per 12.5 Hz frame. Plus `code_predictor` (5-layer mini-Qwen) filling codebooks 1..=15.
//! - `codec` — the 12 Hz speech tokenizer: a Mimi-family RVQ encoder (voice cloning)
//!   and the custom Qwen3TTSTokenizerV2 decoder (codes → 24 kHz waveform).
//! - `speaker_encoder` — ECAPA-TDNN producing a raw 2048-d speaker embedding for cloning.
pub mod codec;
pub mod config;
pub mod quantized_talker;
pub mod rope_fused;
pub mod speaker_encoder;
pub mod talker;
