//! Realtime voice dialogue: mic -> FireRedVAD (endpointing) -> Qwen3-ASR ->
//! MiniCPM5 (sentence-streamed) -> MOSS-TTS per sentence -> speaker. All five
//! models load once on a single Metal device and stay resident.
//!
//! Usage:
//!     tiny-cpm live <vad-dir> <qwen3asr-dir> <minicpm5.gguf | bf16-dir> \
//!         <tokenizer.json> <moss-dir> <codec-dir> \
//!         [--input <wav>] [--output <wav>] [--max-tokens N]
//!
//! - `--input`: simulation mode — frames come from a wav file (fed as fast as
//!   they process, no realtime pacing) instead of the microphone. 1s of
//!   silence is appended so the final utterance endpoints.
//! - `--output`: in simulation mode, write the synthesized reply audio
//!   (stereo 48kHz wav) here instead of playing it.
//! - `--max-tokens`: LLM reply cap (default 256).
//!
//! stdout carries the conversational payload (`you:`/`tiny:` lines); all
//! diagnostics and per-stage timings go to stderr.
//!
//! v1 limitations: no barge-in (TTS keeps playing while the mic stays live,
//! so speaker output may be picked up as a new "utterance" — use headphones);
//! turns are STATELESS — each utterance starts a fresh single-turn ChatML
//! conversation, the LLM does not remember previous turns; while a turn is
//! being processed, mic frames queue up in an unbounded channel (stream
//! continuity is preserved, but turn latency is always >= processing time);
//! Ctrl-C exits immediately without draining playback.

use std::io::Write;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::{
    exec::{chat, moss_tts::MossEngine, qwen3_asr::Qwen3AsrEngine},
    models::fire_red_vad::vad::FireRedVad,
    utils::{
        audio_utils::{load_audio_with_resample, save_wav},
        live_audio::{MicCapture, Speaker, VAD_FRAME_SAMPLES, VAD_SAMPLE_RATE},
    },
};

/// ASR decode cap (matches the `asr qwen3` default).
const ASR_MAX_TOKENS: usize = 512;
/// Default LLM reply cap.
const DEFAULT_MAX_TOKENS: usize = 256;
/// MOSS codec-frame cap per reply sentence (300 frames @ 12.5 fps ~= 24 s).
const MOSS_MAX_FRAMES: usize = 300;
/// TTS streaming granularity: decode+emit every this many codec frames
/// (25 frames @ 12.5 fps ~= 2 s of audio per chunk).
const TTS_CHUNK_FRAMES: usize = 25;
/// Skip VAD segments shorter than this (0.5 s at 16kHz).
const MIN_SEGMENT_SAMPLES: usize = 8000;
/// Silence appended in simulation mode so the last utterance endpoints.
const SIM_TRAILING_SILENCE_SEC: usize = 1;

/// Splits a streamed text into sentences at [.!?;] and their full-width
/// variants (plus …). Sentences include their terminator.
struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    const TERMINATORS: [char; 8] = ['。', '！', '？', '；', '.', '!', '?', ';'];

    fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Append a streamed delta; returns any sentences completed by it.
    fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        let mut boundaries = Vec::new(); // byte index just past each terminator
        let chars: Vec<(usize, char)> = self.buf.char_indices().collect();
        for (idx, &(i, c)) in chars.iter().enumerate() {
            if !Self::TERMINATORS.contains(&c) {
                continue;
            }
            // CJK terminators always split. ASCII ones only when followed by
            // whitespace or end-of-buffer, so "3.5" and "e.g." don't split.
            let next = chars.get(idx + 1).map(|&(_, nc)| nc);
            if c.is_ascii() && next.is_some_and(|nc| !nc.is_whitespace()) {
                continue;
            }
            boundaries.push(i + c.len_utf8());
        }
        if let Some(&last) = boundaries.last() {
            let completed: String = self.buf.drain(..last).collect();
            let mut start = 0usize;
            for end in boundaries {
                let sentence = completed[start..end].trim();
                // Drop punctuation-only pieces ("...", "?!") — synthesizing
                // them would cost seconds of compute for an audible blip.
                if sentence.chars().any(char::is_alphanumeric) {
                    out.push(sentence.to_string());
                }
                start = end;
            }
        }
        out
    }

    /// Remainder without a terminator (call at end of generation).
    fn flush(&mut self) -> Option<String> {
        let rest = self.buf.trim();
        // Same rule as push(): punctuation-only remainders ("**") are junk.
        if rest.is_empty() || !rest.chars().any(char::is_alphanumeric) {
            None
        } else {
            Some(rest.to_string())
        }
    }
}

/// Synthesize one sentence, streaming audio chunks to the speaker (or the
/// simulation buffer) as they are generated instead of waiting for the whole
/// sentence. Errors are logged, not propagated (the show goes on).
fn synth_and_play(
    tts: &mut MossEngine,
    speaker: &Option<Speaker>,
    sim_out: &mut Vec<f32>,
    tts_sr: usize,
    n: usize,
    text: &str,
) {
    let mut on_chunk = |pcm: Vec<f32>| {
        if let Some(speaker) = speaker {
            if let Err(e) = speaker.push(&pcm, tts_sr, 2) {
                eprintln!("playback error: {e}");
            }
        } else {
            sim_out.extend_from_slice(&pcm);
        }
    };
    match tts.synthesize_pcm_stream(text, MOSS_MAX_FRAMES, TTS_CHUNK_FRAMES, &mut on_chunk) {
        Ok(stats) => eprintln!(
            "tts[{n}]: {} frames in {:.2}s (ttft {:.2}s)",
            stats.frames,
            stats.total.as_secs_f64(),
            stats.ttft.as_secs_f64()
        ),
        Err(e) => eprintln!("tts[{n}] error: {e}; continuing"),
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let usage = "usage: tiny-cpm live <vad-dir> <qwen3asr-dir> <minicpm5.gguf | bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> [--input <wav>] [--output <wav>] [--max-tokens N]";
    if args.len() < 6 {
        bail!(usage);
    }
    let vad_dir = &args[0];
    let asr_dir = &args[1];
    let minicpm_path = &args[2];
    let tok_path = &args[3];
    let moss_dir = &args[4];
    let codec_dir = &args[5];
    let mut input_wav: Option<String> = None;
    let mut output_wav: Option<String> = None;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut i = 6;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input_wav = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--input requires a wav path. {usage}"))?
                        .clone(),
                );
            }
            "--output" => {
                i += 1;
                output_wav = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--output requires a wav path. {usage}"))?
                        .clone(),
                );
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--max-tokens requires a count. {usage}"))?
                    .parse()
                    .map_err(|_| anyhow!("--max-tokens must be a positive integer. {usage}"))?;
            }
            other => bail!("unknown option {other}. {usage}"),
        }
        i += 1;
    }
    let sim_mode = input_wav.is_some();
    if !sim_mode && output_wav.is_some() {
        bail!("--output is only meaningful with --input (simulation mode)");
    }

    // --- load all five models on one Metal device ---
    let device = Device::new_metal(0)?;

    let t = Instant::now();
    // VAD runs on CPU: it is tiny (2MB), and keeping it off Metal leaves the
    // GPU free for TTS generation (measured ~1.5x faster MOSS in-process).
    let mut vad = FireRedVad::init(vad_dir, Some(&Device::Cpu), None)?;
    eprintln!(
        "loaded FireRedVAD (cpu) in {:.2}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let mut asr = Qwen3AsrEngine::load(asr_dir, &device)?;
    eprintln!("loaded Qwen3-ASR in {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut llm = chat::load_model(minicpm_path, &device)?;
    eprintln!("loaded MiniCPM5 in {:.2}s", t.elapsed().as_secs_f64());
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow!("tokenizer: {e}"))?;

    let t = Instant::now();
    let mut tts = MossEngine::load(moss_dir, codec_dir, &device)?;
    eprintln!("loaded MOSS-TTS in {:.2}s", t.elapsed().as_secs_f64());
    let tts_sr = tts.sample_rate();

    // --- frame source + audio sink ---
    let mic = if sim_mode {
        None
    } else {
        Some(MicCapture::start()?)
    };
    let sim_samples: Vec<f32> = if let Some(wav) = &input_wav {
        let audio = load_audio_with_resample(wav, &device, Some(VAD_SAMPLE_RATE), Some(1))?;
        let mut samples = audio.squeeze(0)?.to_vec1::<f32>()?;
        samples.extend(std::iter::repeat_n(
            0.0,
            SIM_TRAILING_SILENCE_SEC * VAD_SAMPLE_RATE,
        ));
        eprintln!(
            "sim input: {} (+{SIM_TRAILING_SILENCE_SEC}s trailing silence)",
            wav
        );
        samples
    } else {
        Vec::new()
    };
    let speaker = if sim_mode {
        None
    } else {
        Some(Speaker::start()?)
    };
    // Simulation mode accumulates the reply PCM (interleaved stereo) here.
    let mut sim_out: Vec<f32> = Vec::new();

    eprintln!(
        "live: {} — listening (Ctrl-C to quit)",
        if sim_mode {
            "simulation mode"
        } else {
            "mic mode (no barge-in; use headphones)"
        }
    );

    // --- main loop: VAD frames -> utterance -> ASR -> LLM -> TTS ---
    let mut sim_pos = 0usize;
    let mut turn = 0usize;
    // Mic-ducking state: when Some, we are dropping frames during playback.
    let mut duck_since: Option<Instant> = None;
    // Heartbeat: log mic frame RMS once per second so capture problems are
    // visible (silent mic vs VAD not triggering vs ducking active).
    let mut frame_count = 0usize;
    loop {
        let frame = if let Some(mic) = &mic {
            mic.next_frame()?
        } else {
            if sim_pos >= sim_samples.len() {
                break; // end of --input file: exit cleanly
            }
            let end = (sim_pos + VAD_FRAME_SAMPLES).min(sim_samples.len());
            let mut frame = sim_samples[sim_pos..end].to_vec();
            frame.resize(VAD_FRAME_SAMPLES, 0.0); // pad final partial frame
            sim_pos = end;
            frame
        };
        frame_count += 1;
        if frame_count % 40 == 0 {
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
            eprintln!(
                "heartbeat: mic rms {rms:.4} ({})",
                if duck_since.is_some() {
                    "ducking"
                } else {
                    "listening"
                }
            );
        }
        // Mic ducking: while TTS audio is playing back, drop input frames —
        // otherwise the mic hears the speaker and the system echo-loops
        // (talks to itself). v1 has no barge-in, so input during playback is
        // useless anyway. A 0.2s tail covers room echo after the queue drains.
        // Failsafe: if the queue stops draining for 20s (dead output stream),
        // clear it and resume listening instead of muting forever.
        if let Some(speaker) = &speaker {
            let queued = speaker.queued_seconds();
            if queued > 0.2 {
                let since = duck_since.get_or_insert_with(|| {
                    eprintln!("ducking: mic muted during playback ({queued:.1}s queued)");
                    Instant::now()
                });
                if since.elapsed() < std::time::Duration::from_secs(20) {
                    continue;
                }
                eprintln!(
                    "warning: playback queue stuck at {queued:.1}s for 20s; clearing and resuming mic (audio output device issue?)"
                );
                speaker.clear();
                duck_since = None;
            } else if duck_since.take().is_some() {
                eprintln!("ducking: playback done, mic live");
            }
        }
        let Some(result) = vad.detect_frame_f32(frame, 1, Some(VAD_SAMPLE_RATE))? else {
            continue;
        };
        let Some(segment) = result.orig_audio else {
            continue;
        };
        // orig_audio holds the raw 16kHz mono samples as fed (the x32768
        // scaling inside detect_frame is only used for feature extraction).
        let samples = segment.to_vec1::<f32>()?;
        if samples.len() < MIN_SEGMENT_SAMPLES {
            eprintln!(
                "turn skipped: segment too short ({:.2}s)",
                samples.len() as f32 / VAD_SAMPLE_RATE as f32
            );
            continue;
        }
        turn += 1;
        eprintln!(
            "=== turn {turn}: utterance {:.2}s ===",
            samples.len() as f32 / VAD_SAMPLE_RATE as f32
        );

        // ASR
        let t = Instant::now();
        let transcript = asr.transcribe_samples(&samples, ASR_MAX_TOKENS)?;
        let asr_secs = t.elapsed().as_secs_f64();
        if transcript.trim().is_empty() {
            eprintln!("turn {turn}: empty transcript, skipping (asr {asr_secs:.2}s)");
            continue;
        }
        println!("you: {transcript}");
        let _ = std::io::stdout().flush();

        // LLM, streamed; sentences are synthesized as soon as they complete.
        let t_llm = Instant::now();
        let mut splitter = SentenceSplitter::new();
        let mut first_sentence_at: Option<f64> = None;
        let mut n_sentences = 0usize;
        let mut sink = |delta: &str| {
            for sentence in splitter.push(delta) {
                if first_sentence_at.is_none() {
                    first_sentence_at = Some(t_llm.elapsed().as_secs_f64());
                }
                n_sentences += 1;
                println!("tiny: {sentence}");
                let _ = std::io::stdout().flush();
                synth_and_play(
                    &mut tts,
                    &speaker,
                    &mut sim_out,
                    tts_sr,
                    n_sentences,
                    &sentence,
                );
            }
        };
        let llm_stats = match chat::generate_reply_with_system(
            &mut llm,
            &tokenizer,
            &device,
            // MiniCPM5 is a reasoning model and sometimes emits its inner
            // monologue WITHOUT <think> tags; this keeps the spoken reply short.
            Some(
                "你是一个语音助手的回复模块。直接用一两句简短的口语回答用户，禁止输出思考过程、分析、复述用户问题或任何 markdown 格式。",
            ),
            &transcript,
            true, // no_think: append an empty think block (enable_thinking=false)
            max_tokens,
            &mut sink,
        ) {
            Ok(stats) => stats,
            // The model is stateless per call, so a transient decode error
            // only loses this turn — log and keep the session alive.
            Err(e) => {
                eprintln!("turn {turn}: llm error: {e}; skipping turn");
                continue;
            }
        };
        drop(sink);
        if let Some(rest) = splitter.flush() {
            if first_sentence_at.is_none() {
                first_sentence_at = Some(t_llm.elapsed().as_secs_f64());
            }
            n_sentences += 1;
            println!("tiny: {rest}");
            let _ = std::io::stdout().flush();
            synth_and_play(&mut tts, &speaker, &mut sim_out, tts_sr, n_sentences, &rest);
        }
        if n_sentences == 0 {
            eprintln!("turn {turn}: empty reply, skipping");
            continue;
        }
        eprintln!(
            "turn {turn} timings: asr {asr_secs:.2}s, llm first sentence {:.2}s ({} tokens, {:.1} tok/s), {n_sentences} sentence(s)",
            first_sentence_at.unwrap_or_default(),
            llm_stats.tokens,
            llm_stats.tokens as f64 / llm_stats.decode.as_secs_f64().max(1e-9),
        );
    }

    // --- simulation mode: write the accumulated reply audio ---
    if let Some(out_path) = &output_wav {
        if sim_out.is_empty() {
            eprintln!("no reply audio synthesized; not writing {out_path}");
        } else {
            let frames = sim_out.len() / 2;
            let interleaved = Tensor::new(sim_out.as_slice(), &Device::Cpu)?;
            let stereo = interleaved.reshape((frames, 2))?.t()?;
            save_wav(&stereo, out_path, 2, tts_sr as u32)?;
            eprintln!(
                "wrote {out_path} ({:.2}s stereo @ {} Hz)",
                frames as f64 / tts_sr as f64,
                tts_sr
            );
        }
    }
    if let Some(speaker) = &speaker {
        eprintln!(
            "{:.2}s of audio left in the playback queue",
            speaker.queued_seconds()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SentenceSplitter;

    #[test]
    fn cjk_terminators_split() {
        let mut s = SentenceSplitter::new();
        let out = s.push("你好。世界！");
        assert_eq!(out, vec!["你好。", "世界！"]);
    }

    #[test]
    fn ascii_dot_inside_number_does_not_split() {
        let mut s = SentenceSplitter::new();
        let out = s.push("版本是 3.5 秒。");
        assert_eq!(out, vec!["版本是 3.5 秒。"]);
    }

    #[test]
    fn ellipsis_then_new_sentence() {
        // The '.' after whitespace ends the ellipsis sentence (no junk
        // punctuation-only pieces, and "3.5"-style dots never split).
        let mut s = SentenceSplitter::new();
        let out = s.push("Hmm... let me think. ");
        assert_eq!(out, vec!["Hmm...", "let me think."]);
    }

    #[test]
    fn punctuation_only_pieces_dropped() {
        let mut s = SentenceSplitter::new();
        let out = s.push("好！？！");
        assert_eq!(out, vec!["好！"]);
    }

    #[test]
    fn flush_returns_remainder() {
        let mut s = SentenceSplitter::new();
        assert!(s.push("还没说完").is_empty());
        assert_eq!(s.flush().as_deref(), Some("还没说完"));
    }
}
