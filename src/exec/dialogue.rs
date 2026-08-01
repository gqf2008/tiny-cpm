//! Dialogue pipeline: chains Fun-ASR-Nano (ASR) -> MiniCPM5-1B (LLM chat) ->
//! MOSS-TTS-Nano (TTS) in one process. All three models are loaded once on a
//! single Metal device and stay resident for the whole run.
//!
//! Usage:
//!     tiny-cpm dialogue <funasr-dir> <bf16-dir> <tokenizer.json> \
//!         <moss-dir> <codec-dir> <input.wav> <output.wav> [max_tokens]
//!
//! `max_tokens` caps the LLM reply (default 256). stdout carries only the
//! conversational payload (transcript + reply); all diagnostics and the
//! per-stage latency summary go to stderr.
//!
//! Diagnostic probes (env vars, for perf investigation only):
//! - `DIALOGUE_PROBE=1` runs a fixed MOSS synthesis probe (steady ms/frame ×3)
//!   after loads / after ASR / after LLM.
//! - `DIALOGUE_PROBE_ONLY=1` exits after the after-loads probe; combine with
//!   `PROBE_NO_FUNASR=1` / `PROBE_NO_LLM=1` to measure MOSS with only a subset
//!   of the other models resident (skip flags have no effect otherwise).

use std::time::Instant;

use anyhow::{Result, bail};
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::exec::{chat, fun_asr_nano::FunAsrEngine, moss_tts::MossEngine};

/// ASR decode cap (matches the `asr funasr` default).
const ASR_MAX_TOKENS: usize = 512;
/// Default LLM reply cap.
const DEFAULT_MAX_TOKENS: usize = 256;
/// MOSS codec-frame cap (300 frames @ 12.5 fps ~= 24 s of audio).
const MOSS_MAX_FRAMES: usize = 300;

pub fn run(args: &[String]) -> Result<()> {
    let usage = "usage: tiny-cpm dialogue <funasr-dir> <bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> <input.wav> <output.wav> [max_tokens]";
    if args.len() < 7 {
        bail!(usage);
    }
    let funasr_dir = &args[0];
    let minicpm_path = &args[1];
    let tok_path = &args[2];
    let moss_dir = &args[3];
    let codec_dir = &args[4];
    let input_wav = &args[5];
    let output_wav = &args[6];
    let max_tokens = args
        .get(7)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid max_tokens: {e}"))?
        .unwrap_or(DEFAULT_MAX_TOKENS);

    // --- load all three models on one Metal device ---
    let device = Device::new_metal(0)?;

    // DIALOGUE_PROBE=1: run a fixed MOSS probe synthesis at three points to
    // isolate what slows TTS down in-process (residency vs ASR run vs LLM run).
    // DIALOGUE_PROBE_ONLY=1 exits right after the first probe; combined with
    // PROBE_NO_FUNASR=1 / PROBE_NO_LLM=1 it measures MOSS speed with only a
    // subset of the other models resident.
    let probe = std::env::var("DIALOGUE_PROBE").as_deref() == Ok("1");
    let probe_only = probe && std::env::var("DIALOGUE_PROBE_ONLY").is_ok();
    let skip_funasr = probe_only && std::env::var("PROBE_NO_FUNASR").is_ok();
    let skip_llm = probe_only && std::env::var("PROBE_NO_LLM").is_ok();
    if !probe
        && (std::env::var("DIALOGUE_PROBE_ONLY").is_ok()
            || std::env::var("PROBE_NO_FUNASR").is_ok()
            || std::env::var("PROBE_NO_LLM").is_ok())
    {
        eprintln!("warning: probe env vars have no effect without DIALOGUE_PROBE=1");
    } else if !probe_only
        && (std::env::var("PROBE_NO_FUNASR").is_ok() || std::env::var("PROBE_NO_LLM").is_ok())
    {
        eprintln!(
            "warning: PROBE_NO_FUNASR/PROBE_NO_LLM have no effect without DIALOGUE_PROBE_ONLY=1"
        );
    }

    let t = Instant::now();
    let mut asr = if skip_funasr {
        None
    } else {
        Some(FunAsrEngine::load(funasr_dir, &device)?)
    };
    let load_funasr = t.elapsed();

    let t = Instant::now();
    let mut llm = if skip_llm {
        None
    } else {
        Some(chat::load_model(minicpm_path, &device)?)
    };
    let load_minicpm5 = t.elapsed();

    let tokenizer =
        Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let t = Instant::now();
    let mut tts = MossEngine::load(moss_dir, codec_dir, &device)?;
    let load_moss = t.elapsed();

    let mut run_probe = |tts: &mut MossEngine, tag: &str| -> Result<()> {
        // Repeat 3×: frame counts vary stochastically, so report steady-state
        // per-frame time (total - ttft) / (frames - 1), which is comparable
        // across runs of different lengths.
        for i in 1..=3 {
            let stats = tts.synthesize(
                "边缘计算正在改变人工智能的部署方式。",
                "/tmp/dialogue_probe.wav",
                60,
                None,
            )?;
            let steady_ms = (stats.total - stats.ttft).as_secs_f64() * 1000.0
                / (stats.frames.max(2) - 1) as f64;
            eprintln!(
                "probe[{tag}#{i}]: {} frames in {:.2}s (steady {:.1} ms/frame, ttft {:.3}s)",
                stats.frames,
                stats.total.as_secs_f64(),
                steady_ms,
                stats.ttft.as_secs_f64()
            );
        }
        Ok(())
    };
    if probe {
        run_probe(&mut tts, "after-loads")?;
    }
    if probe_only {
        return Ok(());
    }
    let asr = asr.as_mut().expect("asr engine loaded");
    let llm = llm.as_mut().expect("llm loaded");

    // --- ASR: input.wav -> transcript ---
    let t = Instant::now();
    let transcript = asr.transcribe(input_wav, ASR_MAX_TOKENS)?;
    let asr_secs = t.elapsed().as_secs_f64();
    eprintln!("transcript: {transcript}");
    if transcript.trim().is_empty() {
        bail!("ASR produced an empty transcript for {input_wav}; aborting before LLM/TTS");
    }
    if probe {
        run_probe(&mut tts, "after-asr")?;
    }

    // --- LLM: transcript -> reply (think tags stripped, not streamed) ---
    let mut sink = |_s: &str| {};
    let llm_stats =
        chat::generate_reply(llm, &tokenizer, &device, &transcript, max_tokens, &mut sink)?;
    let reply = llm_stats.text;
    eprintln!("reply: {reply}");
    if reply.trim().is_empty() {
        bail!("LLM produced an empty reply (after stripping think tags); aborting before TTS");
    }
    if probe {
        run_probe(&mut tts, "after-llm")?;
    }

    // --- TTS: reply -> output.wav ---
    let tts_stats = tts.synthesize(&reply, output_wav, MOSS_MAX_FRAMES, None)?;

    // --- payload to stdout, latency summary to stderr ---
    println!("transcript: {transcript}");
    println!("reply: {reply}");

    let llm_secs = llm_stats.decode.as_secs_f64();
    let tts_secs = tts_stats.total.as_secs_f64();
    eprintln!("=== stage latency ===");
    eprintln!(
        "load: funasr {:.1}s, minicpm5 {:.1}s, moss {:.1}s",
        load_funasr.as_secs_f64(),
        load_minicpm5.as_secs_f64(),
        load_moss.as_secs_f64()
    );
    eprintln!(
        "asr: {asr_secs:.2}s (transcript: {} chars)",
        transcript.chars().count()
    );
    eprintln!(
        "llm: {llm_secs:.2}s ({} tokens, {:.1} tok/s)",
        llm_stats.tokens,
        llm_stats.tokens as f64 / llm_secs
    );
    eprintln!(
        "tts: {tts_secs:.2}s ({} frames, TTFT {:.2}s, codec decode {:.2}s)",
        tts_stats.frames,
        tts_stats.ttft.as_secs_f64(),
        tts_stats.codec_decode.as_secs_f64()
    );
    eprintln!("total inference: {:.2}s", asr_secs + llm_secs + tts_secs);
    Ok(())
}
