//! CosyVoice3 (Fun-CosyVoice3-0.5B-2512) TTS, ported from CrispASR's C++/ggml
//! implementation (github.com/CrispStrobe/CrispASR, src/cosyvoice3_tts.cpp).
//!
//! Pipeline: text (Qwen2 BPE) -> LM (Qwen2-0.5B AR, speech tokens) -> Flow
//! (DiT-CFM, Euler+CFG, mel) -> HiFT (NSF + iSTFT, 24kHz waveform). Voice
//! cloning adds s3tok (ref wav -> speech tokens) and CAMPPlus (ref wav ->
//! speaker embedding).

pub mod campplus;
pub mod flow;
pub mod hift;
pub mod lm;
pub mod pipeline;
pub mod s3tok;
