//! tiny-cpm library: on-device (Apple Metal) inference for MiniCPM5 chat,
//! Fun-ASR / Qwen3-ASR, and VoxCPM2 / MOSS-TTS-Nano / CosyVoice3 — plus
//! FireRedVAD. The `tiny-cpm` binary (`src/main.rs`) is the CLI front-end; the
//! `web` crate reuses these engines for the WebSocket voice-dialogue server.
//!
//! See `CLAUDE.md` / `AGENTS.md` for the model list and architecture.

pub mod common;
pub mod exec;
pub mod models;
pub mod position_embed;
pub mod quantized_minicpm5;
pub mod token_output_stream;
pub mod tokenizer;
pub mod utils;
