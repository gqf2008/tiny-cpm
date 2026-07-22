//! Qwen3-ASR, ported from aha (github.com/jhqxxx/aha) src/models/qwen3_asr/mod.rs.
//! (aha's generate.rs is not ported — it is rocket/minijinja-coupled; the CLI
//! driver in `crate::exec::qwen3_asr` fills its role.)

pub mod config;
pub mod model;
pub mod processor;
