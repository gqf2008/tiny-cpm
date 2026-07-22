//! Fun-ASR-Nano, ported from aha (github.com/jhqxxx/aha) `src/models/fun_asr_nano/`.
//!
//! aha's `generate.rs` is not ported (it is coupled to rocket/params); the
//! load + decode loop lives in `crate::exec::fun_asr_nano` instead.
pub mod config;
pub mod model;
pub mod processor;
