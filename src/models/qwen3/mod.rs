//! Qwen3 decoder (shared by both ASR models), ported from aha `src/models/qwen3/`.
//!
//! Ported from aha (github.com/jhqxxx/aha) src/models/qwen3/mod.rs.
//! `generate.rs` (server-coupled generation: chat_template/params/server) is dropped.

pub mod config;
pub mod model;
