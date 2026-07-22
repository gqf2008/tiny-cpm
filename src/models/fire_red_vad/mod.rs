//! FireRedVAD, ported from aha (github.com/jhqxxx/aha) `src/models/fire_red_vad/`.
//!
//! Only the safetensors checkpoint layout (model.safetensors + cmvn.json, e.g.
//! FireRedVAD-Stream-VAD) is supported, same as aha: `vad.rs` loads weights via
//! `find_type_files(path, "safetensors")` and `processor.rs` reads `cmvn.json`.
//! Torch `.pth.tar` + `cmvn.ark` checkpoints (FireRedVAD-VAD / FireRedVAD-AED)
//! are rejected by the CLI driver in `crate::exec::vad`.
pub mod config;
pub mod model;
pub mod processor;
pub mod vad;
