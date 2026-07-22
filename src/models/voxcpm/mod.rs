//! VoxCPM2, ported from aha `src/models/voxcpm/` (+ processor/pipeline logic from `voxcpm_refact/`).
//!
//! - `config`, `tokenizer`, `minicpm4`, `model`, `audio_vae` are near-verbatim ports of
//!   aha's `src/models/voxcpm/` files.
//! - `processor` is aha's `src/models/voxcpm_refact/processor.rs`.
//! - `pipeline` merges aha's `voxcpm_refact/model.rs` (`VoxCPMModelRefact`, non-streaming
//!   inference only) with the rocket-free parts of `voxcpm_refact/generate.rs`
//!   (`VoxCPMGenerate`: weight loading + inference entry points).

pub mod audio_vae;
pub mod config;
pub mod minicpm4;
pub mod model;
pub mod pipeline;
pub mod processor;
pub mod tokenizer;
