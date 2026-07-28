//! FireRedVAD driver. Usage: tiny-cpm vad <model-dir> <audio-file>
//!
//! Ported from aha (github.com/jhqxxx/aha): CLI shape from tests/test_fire_red_vad.rs
//! (init -> detect_file -> print segments). Speech segments go to stdout, one per
//! line as `start_seconds end_seconds`; diagnostics go to stderr.

use std::{path::Path, time::Instant};

use anyhow::{Result, bail};
use candle_core::Device;

use crate::models::fire_red_vad::vad::FireRedVad;

pub fn run(args: &[String]) -> Result<()> {
    let usage = "usage: tiny-cpm vad <model-dir> <audio-file>";
    let Some(model_dir) = args.first() else {
        bail!(usage.to_string());
    };
    let Some(audio_file) = args.get(1) else {
        bail!(usage.to_string());
    };

    // aha's FireRedVad only loads safetensors weights + cmvn.json. The torch
    // checkpoints (.pth.tar + cmvn.ark, e.g. FireRedVAD-VAD / FireRedVAD-AED)
    // are not loadable; fail clearly instead of panicking in CMVN::new.
    let has_cmvn = Path::new(model_dir).join("cmvn.json").is_file();
    let has_safetensors = std::fs::read_dir(model_dir)
        .map(|mut rd| {
            rd.any(|e| {
                e.as_ref()
                    .is_ok_and(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            })
        })
        .unwrap_or(false);
    if !has_cmvn || !has_safetensors {
        bail!(
            "only the safetensors layout (model.safetensors + cmvn.json, e.g. FireRedVAD-Stream-VAD) is supported in {model_dir}; torch .pth.tar + cmvn.ark checkpoints are not loadable"
        );
    }

    // TINY_CPM_DEVICE=cpu forces CPU inference (default: Metal).
    let device = if std::env::var("TINY_CPM_DEVICE").as_deref() == Ok("cpu") {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    eprintln!("device: {device:?}");

    let start = Instant::now();
    let vad = FireRedVad::init(model_dir, Some(&device), None, None)?;
    eprintln!("loaded model in {:.2?}", start.elapsed());

    let res = vad.detect_file(audio_file)?;
    eprintln!(
        "model: {} mode: {} dur: {:.2}s segments: {}",
        res.model_name,
        res.mode,
        res.dur,
        res.timestamps.len()
    );
    for (start, end) in &res.timestamps {
        println!("{start:.2} {end:.2}");
    }
    Ok(())
}
