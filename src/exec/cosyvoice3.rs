//! CosyVoice3 (Fun-CosyVoice3-0.5B-2512) TTS driver. Usage:
//!   tiny-cpm tts cosyvoice3 <model-dir> "<text>" <out.wav> [--voice <name>] [--ref <ref.wav>] [--ref-text "<text>"] [--steps N] [--max-tokens N]
//!
//! Baked voices come from voices.gguf in the model dir (default
//! `zero_shot`); `--ref` runs zero-shot cloning from a reference wav instead
//! (s3tok + CAMPPlus GGUFs required; `--ref-text` is the transcript of the
//! ref wav and is REQUIRED — CrispASR refuses zero-shot cloning without it).
//! stdout stays empty: the WAV goes to
//! the given path, diagnostics to stderr.

use std::time::Instant;

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};

use crate::models::cosyvoice3::pipeline::{CosyVoice3Pipeline, SAMPLE_RATE};
use crate::utils::audio_utils::save_wav_mono;

const USAGE: &str = "usage: tiny-cpm tts cosyvoice3 <model-dir> \"<text>\" <out.wav> [--voice <name>] [--ref <ref.wav>] [--ref-text \"<text>\"] [--steps N] [--max-tokens N]";

pub fn run(args: &[String]) -> Result<()> {
    let mut positional: Vec<&str> = Vec::new();
    let mut voice_name = "zero_shot".to_string();
    let mut ref_wav: Option<String> = None;
    let mut ref_text: Option<String> = None;
    let mut steps: usize = 4; // 4 Euler steps: ~30% faster than the reference
    // default of 6 with ASR-verified quality parity
    // (use --steps 6 for the reference default)
    let mut max_tokens: usize = 0; // 0 -> 20 * n_text_ids, min 16
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--voice" => {
                i += 1;
                voice_name = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--voice requires a <name>"))?
                    .clone();
            }
            "--ref" => {
                i += 1;
                ref_wav = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--ref requires a <ref.wav> path"))?
                        .clone(),
                );
            }
            "--ref-text" => {
                i += 1;
                ref_text = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--ref-text requires a value"))?
                        .clone(),
                );
            }
            "--steps" => {
                i += 1;
                steps = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--steps requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--steps must be a positive integer"))?;
                if steps == 0 {
                    return Err(anyhow!("--steps must be a positive integer"));
                }
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-tokens requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--max-tokens must be a non-negative integer"))?;
            }
            other => positional.push(other),
        }
        i += 1;
    }
    let [model_dir, text, out_wav] = positional.as_slice() else {
        return Err(anyhow!(USAGE));
    };
    if ref_wav.is_none() && ref_text.is_some() {
        return Err(anyhow!("--ref-text only makes sense together with --ref"));
    }
    // CrispASR refuses zero-shot cloning without the ref transcript
    // (cosyvoice3_tts.cpp:5663-5667).
    if ref_wav.is_some() && ref_text.is_none() {
        return Err(anyhow!(
            "--ref requires --ref-text (the transcript of the reference wav)"
        ));
    }

    // TINY_CPM_DEVICE=cpu forces CPU inference (default: Metal).
    let device = if std::env::var("TINY_CPM_DEVICE").as_deref() == Ok("cpu") {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    eprintln!("device: {device:?}");

    let t0 = Instant::now();
    let mut pipe = CosyVoice3Pipeline::load(model_dir, &device)?;
    eprintln!("cosyvoice3: pipeline loaded in {:.2?}", t0.elapsed());

    // Fixed seed for reproducible output (LM RAS sampling + flow init noise).
    const SEED: u64 = 42;
    const CFG: f64 = 0.7; // CrispASR cfm_inference_cfg_rate default

    let voice = match &ref_wav {
        Some(path) => {
            if voice_name != "zero_shot" {
                eprintln!("cosyvoice3: note: --ref overrides --voice");
            }
            let t0 = Instant::now();
            let v = pipe.clone_voice(path, ref_text.as_deref().unwrap_or(""))?;
            eprintln!(
                "cosyvoice3: voice extracted from {path} in {:.2?}",
                t0.elapsed()
            );
            v
        }
        None => pipe.get_voice(&voice_name)?,
    };
    eprintln!(
        "cosyvoice3: voice '{}' ({} prompt speech tokens, {} ref mel frames)",
        voice.name,
        voice.prompt_speech_tokens.len(),
        voice.ref_mel.dim(0)?
    );

    let t0 = Instant::now();
    let (wav, stats) = pipe.synthesize(text, &voice, max_tokens, steps, CFG, SEED)?;
    let total = t0.elapsed().as_secs_f64();

    let n = wav.len();
    let wav_t = Tensor::from_vec(wav, (1, n), &device)?;
    save_wav_mono(&wav_t, out_wav, SAMPLE_RATE)?;

    eprintln!(
        "cosyvoice3: lm: {} text ids -> {} speech tokens in {:.2}s ({:.1} tok/s)",
        stats.n_text_ids,
        stats.n_gen_tokens,
        stats.lm_secs,
        stats.n_gen_tokens as f64 / stats.lm_secs.max(1e-9)
    );
    eprintln!("cosyvoice3: flow: {steps} steps in {:.2}s", stats.flow_secs);
    eprintln!("cosyvoice3: hift: {:.2}s", stats.hift_secs);
    eprintln!(
        "cosyvoice3: audio: {:.2}s @ {SAMPLE_RATE} Hz ({} mel frames), total synth {:.2}s",
        stats.audio_secs, stats.t_mel_out, total
    );
    eprintln!("cosyvoice3: wrote {out_wav}");
    Ok(())
}
