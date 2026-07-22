//! Ported from aha (github.com/jhqxxx/aha) src/exec/voxcpm.rs (+ tests/test_voxcpm2.rs,
//! which exercises the `VoxCPMGenerateRefact` pipeline ported in `models/voxcpm/pipeline.rs`).
//!
//! VoxCPM2 TTS driver. Usage:
//!   tiny-cpm tts voxcpm <model-dir> "<text>" <out.wav> [--ref <ref.wav>] [--max-len N]
//!
//! `--ref` enables VoxCPM2 reference-audio voice cloning (no prompt text needed).
//! stdout stays empty: the WAV goes to the given path, diagnostics to stderr.

use std::time::Instant;

use anyhow::{Result, anyhow};
use candle_core::Device;

use crate::{models::voxcpm::pipeline::VoxCPMGenerate, utils::audio_utils::save_wav_mono};

pub fn run(args: &[String]) -> Result<()> {
    let mut positional: Vec<&str> = Vec::new();
    let mut ref_wav: Option<String> = None;
    let mut max_len: usize = 1000; // same default as aha's refact `generate_with_prompt_simple`
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ref" => {
                i += 1;
                ref_wav = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--ref requires a <ref.wav> path"))?
                        .clone(),
                );
            }
            "--max-len" => {
                i += 1;
                max_len = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-len requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--max-len must be a positive integer"))?;
                if max_len == 0 {
                    return Err(anyhow!("--max-len must be a positive integer"));
                }
            }
            other => positional.push(other),
        }
        i += 1;
    }
    let [model_dir, text, out_wav] = positional.as_slice() else {
        return Err(anyhow!(
            "usage: tiny-cpm tts voxcpm <model-dir> \"<text>\" <out.wav> [--ref <ref.wav>] [--max-len N]"
        ));
    };

    // Fail fast on --ref with a non-voxcpm2 checkpoint: check the config
    // before the multi-second weight load (supports_reference() is a pure
    // config predicate).
    if ref_wav.is_some() {
        let cfg_path = format!("{model_dir}/config.json");
        let arch = std::fs::read(&cfg_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v.get("architecture")?.as_str().map(str::to_owned));
        let is_voxcpm2 = arch
            .as_deref()
            .is_some_and(|a| a.eq_ignore_ascii_case("voxcpm2"));
        if !is_voxcpm2 {
            return Err(anyhow!(
                "reference mode is only supported with VoxCPM2 models (config architecture must be \"voxcpm2\", got {arch:?} in {cfg_path})"
            ));
        }
    }

    // TINY_CPM_DEVICE=cpu forces CPU inference (default: Metal).
    let device = if std::env::var("TINY_CPM_DEVICE").as_deref() == Ok("cpu") {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    eprintln!("device: {device:?}");
    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(model_dir, &device)?;
    eprintln!("Time elapsed in load model is: {:?}", i_start.elapsed());

    if ref_wav.is_some() && !voxcpm_generate.supports_reference() {
        return Err(anyhow!(
            "reference mode is only supported with VoxCPM2 models (config architecture must be \"voxcpm2\")"
        ));
    }

    let i_start = Instant::now();
    // VoxCPM2 reference mode: reference wav without prompt text.
    let audio = voxcpm_generate.inference(
        text.to_string(),
        None, // prompt_text
        ref_wav,
        2, // min_len
        max_len,
        10,  // inference_timesteps
        2.0, // cfg_value
        false,
        6.0, // retry_badcase_ratio_threshold (unused with retry_badcase = false)
    )?;
    eprintln!("Time elapsed in generate is: {:?}", i_start.elapsed());

    let sample_rate = voxcpm_generate.sample_rate();
    save_wav_mono(&audio, out_wav, sample_rate as u32)?;
    eprintln!("Output saved to: {}", out_wav);

    Ok(())
}
