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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::{
    exec::{chat, moss_tts::MossEngine, qwen3_asr::Qwen3AsrEngine},
    models::fire_red_vad::vad::{FireRedVad, VadOverrides},
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
/// Skip VAD segments shorter than this (default 0.5 s at 16kHz = 8000). Lower
/// to keep briefer utterances (e.g. 0.3 s = 4800). Priority: --min-segment-
/// samples flag > LIVE_MIN_SEGMENT_SAMPLES env > 8000.
fn min_segment_samples(cli: Option<usize>) -> usize {
    cli.or_else(|| {
        std::env::var("LIVE_MIN_SEGMENT_SAMPLES")
            .ok()
            .and_then(|s| s.parse().ok())
    })
    .unwrap_or(8000)
}
/// Minimum peak abs-sample for a segment to be accepted (default 0.0 = off).
/// A single mic can't beamform, but distant speech (a coworker across the
/// desk) has a much lower peak than the user close to the mic — set this just
/// below YOUR speech peak (read from the `segment peak` log line) to drop
/// bystanders. Priority: --min-seg-peak flag > LIVE_MIN_SEG_PEAK env > 0.0.
fn min_seg_peak(cli: Option<f32>) -> f32 {
    cli.or_else(|| {
        std::env::var("LIVE_MIN_SEG_PEAK")
            .ok()
            .and_then(|s| s.parse().ok())
    })
    .unwrap_or(0.0)
}
/// Silence appended in simulation mode so the last utterance endpoints.
const SIM_TRAILING_SILENCE_SEC: usize = 1;
/// Consecutive neural-speech frames required before firing barge-in (hysteresis
/// so a single borderline frame doesn't trigger). Priority: --barge-onset-frames
/// flag > LIVE_BARGE_ONSET_FRAMES env > 3. Barge uses the FireRedVAD neural
/// per-frame is_speech (not RMS) so it can tell the user's speech apart from
/// headphone bleed / breath, which sit at the same RMS as soft speech.
fn barge_onset_frames(cli: Option<usize>) -> usize {
    cli.or_else(|| {
        std::env::var("LIVE_BARGE_ONSET_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
    })
    .unwrap_or(3)
}
/// Minimum mic RMS for a neural-speech frame to count toward barge. The neural
/// VAD can false-positive on near-silence with spiky mics (bone-conduction
/// headsets) — this floor filters rms~0.001 false positives so barge only fires
/// on real (audible) speech. Priority: --min-barge-rms > LIVE_BARGE_RMS > 0.015.
fn min_barge_rms(cli: Option<f32>) -> f32 {
    cli.or_else(|| {
        std::env::var("LIVE_BARGE_RMS")
            .ok()
            .and_then(|s| s.parse().ok())
    })
    .unwrap_or(0.015)
}
/// Linear gain applied to mic samples before VAD/rms. Low-gain mics (bone-
/// conduction headsets) put soft speech at the noise floor (rms ~0.002-0.02),
/// where the neural VAD can't reliably detect it — boost to bring speech up to
/// normal levels (the noise floor stays below the speech threshold). Priority:
/// --mic-gain > LIVE_MIC_GAIN > 1.0.
fn mic_gain(cli: Option<f32>) -> f32 {
    cli.or_else(|| {
        std::env::var("LIVE_MIC_GAIN")
            .ok()
            .and_then(|s| s.parse().ok())
    })
    .unwrap_or(1.0)
}

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
/// sentence. When `barge` is Some and set, the chunk callback returns `false`
/// to abort synthesis mid-sentence and stops pushing. Errors are logged, not
/// propagated (the show goes on).
fn synth_and_play(
    tts: &mut MossEngine,
    speaker: &Option<Arc<Speaker>>,
    sim_out: &mut Vec<f32>,
    tts_sr: usize,
    n: usize,
    text: &str,
    barge: Option<&AtomicBool>,
    ref_codes: Option<&Tensor>,
) {
    let mut on_chunk = |pcm: Vec<f32>| -> bool {
        if let Some(b) = barge
            && b.load(Ordering::Relaxed)
        {
            return false;
        }
        if let Some(speaker) = speaker {
            if let Err(e) = speaker.push(&pcm, tts_sr, 2) {
                eprintln!("playback error: {e}");
            }
        } else {
            sim_out.extend_from_slice(&pcm);
        }
        true
    };
    match tts.synthesize_pcm_stream_with_codes(
        text,
        MOSS_MAX_FRAMES,
        TTS_CHUNK_FRAMES,
        ref_codes,
        &mut on_chunk,
    ) {
        Ok(stats) => eprintln!(
            "tts[{n}]: {} frames in {:.2}s (ttft {:.2}s){}",
            stats.frames,
            stats.total.as_secs_f64(),
            stats.ttft.as_secs_f64(),
            if barge.is_some_and(|b| b.load(Ordering::Relaxed)) {
                " [aborted by barge-in]"
            } else {
                ""
            },
        ),
        Err(e) => eprintln!("tts[{n}] error: {e}; continuing"),
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let usage = "usage: tiny-cpm live <vad-dir> <qwen3asr-dir> <minicpm5.gguf | bf16-dir> <tokenizer.json> <moss-dir> <codec-dir> [--input <wav>] [--output <wav>] [--max-tokens N] [--barge-in] [--ref <wav>] [--vad-speech-threshold <f>] [--vad-min-speech-frame <n>] [--vad-min-silence <n>] [--vad-min-speach-ratio <f>] [--vad-min-speach-frames <n>] [--vad-end-silence-ratio <f>] [--vad-look-back-frames <n>] [--min-segment-samples <n>] [--min-seg-peak <f>] [--barge-onset-frames <n>] [--min-barge-rms <f>] [--mic-gain <f>]";
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
    let mut barge_in = false;
    let mut ref_wav: Option<String> = None;
    // High-frequency VAD/tuning knobs exposed as flags (priority: flag > env >
    // default). See `vad params:` at startup for the effective values.
    let mut vad_end_silence_ratio: Option<f32> = None;
    let mut vad_min_silence: Option<usize> = None;
    let mut vad_speech_threshold: Option<f32> = None;
    let mut vad_min_speech_frame: Option<usize> = None;
    let mut vad_min_speach_ratio: Option<f32> = None;
    let mut vad_min_speach_frames: Option<usize> = None;
    let mut vad_look_back_frames: Option<usize> = None;
    let mut min_seg_peak_cli: Option<f32> = None;
    let mut min_segment_samples_cli: Option<usize> = None;
    let mut barge_onset_cli: Option<usize> = None;
    let mut min_barge_rms_cli: Option<f32> = None;
    let mut mic_gain_cli: Option<f32> = None;
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
            "--barge-in" => barge_in = true,
            "--ref" => {
                i += 1;
                ref_wav = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--ref requires a wav path. {usage}"))?
                        .clone(),
                );
            }
            "--vad-end-silence-ratio" => {
                i += 1;
                vad_end_silence_ratio = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--vad-end-silence-ratio requires a float. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--vad-end-silence-ratio must be a float. {usage}"))?,
                );
            }
            "--vad-min-silence" => {
                i += 1;
                vad_min_silence = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--vad-min-silence requires a count. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--vad-min-silence must be a positive integer. {usage}"))?,
                );
            }
            "--vad-speech-threshold" => {
                i += 1;
                vad_speech_threshold = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--vad-speech-threshold requires a float. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--vad-speech-threshold must be a float. {usage}"))?,
                );
            }
            "--vad-min-speech-frame" => {
                i += 1;
                vad_min_speech_frame = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--vad-min-speech-frame requires a count. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--vad-min-speech-frame must be a positive integer. {usage}"))?,
                );
            }
            "--vad-min-speach-ratio" => {
                i += 1;
                vad_min_speach_ratio = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--vad-min-speach-ratio requires a float. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--vad-min-speach-ratio must be a float. {usage}"))?,
                );
            }
            "--vad-min-speach-frames" => {
                i += 1;
                vad_min_speach_frames = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--vad-min-speach-frames requires a count. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--vad-min-speach-frames must be a positive integer. {usage}"))?,
                );
            }
            "--vad-look-back-frames" => {
                i += 1;
                vad_look_back_frames = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--vad-look-back-frames requires a count. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--vad-look-back-frames must be a positive integer. {usage}"))?,
                );
            }
            "--min-segment-samples" => {
                i += 1;
                min_segment_samples_cli = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--min-segment-samples requires a count. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--min-segment-samples must be a positive integer. {usage}"))?,
                );
            }
            "--min-seg-peak" => {
                i += 1;
                min_seg_peak_cli = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--min-seg-peak requires a float. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--min-seg-peak must be a float. {usage}"))?,
                );
            }
            "--barge-onset-frames" => {
                i += 1;
                barge_onset_cli = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--barge-onset-frames requires a count. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--barge-onset-frames must be a positive integer. {usage}"))?,
                );
            }
            "--min-barge-rms" => {
                i += 1;
                min_barge_rms_cli = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--min-barge-rms requires a float. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--min-barge-rms must be a float. {usage}"))?,
                );
            }
            "--mic-gain" => {
                i += 1;
                mic_gain_cli = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--mic-gain requires a float. {usage}"))?
                        .parse()
                        .map_err(|_| anyhow!("--mic-gain must be a float. {usage}"))?,
                );
            }
            other => bail!("unknown option {other}. {usage}"),
        }
        i += 1;
    }
    let sim_mode = input_wav.is_some();
    if !sim_mode && output_wav.is_some() {
        bail!("--output is only meaningful with --input (simulation mode)");
    }
    if barge_in && sim_mode {
        bail!("--barge-in is only meaningful in mic mode (not with --input)");
    }

    // --- load all five models on one Metal device ---
    let device = Device::new_metal(0)?;

    let t = Instant::now();
    // VAD runs on CPU: it is tiny (2MB), and keeping it off Metal leaves the
    // GPU free for TTS generation (measured ~1.5x faster MOSS in-process).
    let mut vad = FireRedVad::init(
        vad_dir,
        Some(&Device::Cpu),
        None,
        Some(VadOverrides {
            speech_threshold: vad_speech_threshold,
            min_speech_frame: vad_min_speech_frame,
            min_silence_frame: vad_min_silence,
            min_speach_ratio: vad_min_speach_ratio,
            end_silence_ratio: vad_end_silence_ratio,
            min_speach_frames: vad_min_speach_frames,
            look_back_frames: vad_look_back_frames,
        }),
    )?;
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

    // --- reference voice (optional): encode the ref once and reuse the codes
    // for every reply sentence, so cloning doesn't re-encode a long ref per
    // sentence (a 22 s ref is ~4 s of CPU codec encode each). ---
    let ref_codes = if let Some(ref_path) = &ref_wav {
        let t = Instant::now();
        let codes = tts.encode_ref(ref_path)?;
        eprintln!(
            "encoded voice ref {} ({} frames) in {:.2}s",
            ref_path,
            codes.dim(0)?,
            t.elapsed().as_secs_f64()
        );
        Some(codes)
    } else {
        None
    };

    // --- frame source + audio sink ---
    // In barge-in mode the mic is created on the listener thread (cpal's
    // Stream is !Send on macOS, so it must live on the thread that owns it).
    let mic = if sim_mode || barge_in {
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
    let speaker: Option<Arc<Speaker>> = if sim_mode {
        None
    } else {
        Some(Arc::new(Speaker::start()?))
    };
    // Simulation mode accumulates the reply PCM (interleaved stereo) here.
    let mut sim_out: Vec<f32> = Vec::new();

    eprintln!(
        "live: {} — listening (Ctrl-C to quit)",
        if sim_mode {
            "simulation mode"
        } else if barge_in {
            "mic mode, barge-in ON (HEADPHONES REQUIRED — no AEC, speaker echo false-triggers)"
        } else {
            "mic mode (no barge-in; use headphones)"
        }
    );

    // --- barge-in path: a listener thread owns the mic + VAD and runs
    // continuously (no ducking); on speech onset while speaking it sets a
    // shared `barge` flag and clears the speaker queue. The main thread
    // processes utterances and aborts the in-flight LLM/TTS via the flag. ---
    let mic_gain_val = mic_gain(mic_gain_cli);
    if barge_in {
        let speaker_arc = speaker.clone().unwrap();
        // Speaker is !Send (owns the cpal playback stream); hand the listener
        // just the shared queue Arc (Send) so it can clear playback on barge.
        let speaker_queue = speaker_arc.shared_queue();
        let speaking = Arc::new(AtomicBool::new(false));
        let barge = Arc::new(AtomicBool::new(false));
        let (seg_tx, seg_rx) = mpsc::channel::<Vec<f32>>();

        let speaking_l = speaking.clone();
        let barge_l = barge.clone();
        let speaker_q_l = speaker_queue.clone();
        let onset_frames = barge_onset_frames(barge_onset_cli);
        let min_barge_rms_val = min_barge_rms(min_barge_rms_cli);
        eprintln!(
            "barge-in listener: mic_gain={mic_gain_val:.1} min_barge_rms={min_barge_rms_val:.3} onset_frames={onset_frames}"
        );
        std::thread::spawn(move || {
            // cpal::Stream is !Send on macOS — create the mic on this thread
            // and keep it here for the life of the listener.
            let mic = match MicCapture::start() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("barge-in listener: mic capture failed: {e}");
                    return;
                }
            };
            let mut vad = vad;
            let mut onset_count = 0usize;
            let mut frame_count = 0usize;
            let mut speech_frames_since_hb = 0usize;
            let mut max_rms_since_hb = 0.0f32;
            loop {
                let frame = match mic.next_frame() {
                    Ok(f) => f,
                    Err(_) => break,
                };
                frame_count += 1;
                // Apply mic gain (low-gain mics put soft speech at the noise
                // floor; boost before VAD so the neural model sees normal
                // levels).
                let frame: Vec<f32> = if mic_gain_val != 1.0 {
                    frame.iter().map(|s| (s * mic_gain_val).clamp(-1.0, 1.0)).collect()
                } else {
                    frame
                };
                let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
                if rms > max_rms_since_hb {
                    max_rms_since_hb = rms;
                }
                // Neural VAD per frame (sets last_frame_speech, may return a
                // completed segment). detect_frame_f32 consumes `frame`.
                let vad_result = match vad.detect_frame_f32(frame, 1, Some(VAD_SAMPLE_RATE)) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let is_speech = vad.last_frame_speech();
                if is_speech {
                    speech_frames_since_hb += 1;
                }
                // Barge only while audio is actually playing (speaker queue
                // non-empty) AND the neural VAD says this frame is speech AND
                // the mic RMS is above a floor — the neural VAD can false-
                // positive on near-silence with spiky mics (bone-conduction
                // headsets), and the RMS floor filters that.
                let playing = !speaker_q_l.lock().unwrap().is_empty();
                if speaking_l.load(Ordering::Relaxed)
                    && playing
                    && is_speech
                    && rms >= min_barge_rms_val
                {
                    onset_count += 1;
                    if onset_count >= onset_frames && !barge_l.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "barge-in: neural speech onset (rms {rms:.3}) — cancelling reply"
                        );
                        speaker_q_l.lock().unwrap().clear();
                    }
                } else {
                    onset_count = 0;
                }
                // Heartbeat: print this frame's rms + how many of the last 80
                // frames the neural VAD called speech + the max rms in that
                // window. <10/80 with a high max means the mic only captures
                // bursts (not sustained speech) — a hardware limitation no
                // software gain can fix.
                if frame_count % 80 == 0 {
                    eprintln!(
                        "heartbeat: rms {rms:.4} max_rms {max_rms_since_hb:.4} speech {is_speech} ({speech_frames_since_hb}/80 frames) ({})",
                        if playing { "playing" } else { "listening" }
                    );
                    speech_frames_since_hb = 0;
                    max_rms_since_hb = 0.0;
                }
                let Some(result) = vad_result else {
                    continue;
                };
                let Some(segment) = result.orig_audio else {
                    continue;
                };
                let samples = match segment.to_vec1::<f32>() {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if samples.len() < min_segment_samples(min_segment_samples_cli) {
                    continue;
                }
                // Energy gate: drop distant/bystander speech (a coworker across
                // the desk has a much lower peak than the user close to the mic).
                let peak = samples
                    .iter()
                    .map(|s| s.abs())
                    .fold(0.0f32, |a, b| a.max(b));
                let gate = min_seg_peak(min_seg_peak_cli);
                if peak < gate {
                    eprintln!(
                        "segment dropped: peak {peak:.3} < gate {gate:.3} (distant/noise)"
                    );
                    continue;
                }
                eprintln!("segment peak {peak:.3}");
                let _ = seg_tx.send(samples);
            }
        });

        let mut turn = 0usize;
        loop {
            let samples = match seg_rx.recv() {
                Ok(s) => s,
                Err(_) => break,
            };
            // If the listener raised barge during the previous reply, this
            // segment is the interrupting utterance. Clear the flag for the
            // new reply, and flush any audio that snuck into the queue.
            let was_barge = barge.swap(false, Ordering::Relaxed);
            if was_barge {
                speaker_arc.clear();
            }
            speaking.store(true, Ordering::Relaxed);
            turn += 1;
            eprintln!(
                "=== turn {turn}: utterance {:.2}s{} ===",
                samples.len() as f32 / VAD_SAMPLE_RATE as f32,
                if was_barge { " (barge-in)" } else { "" }
            );

            // ASR
            let t = Instant::now();
            let transcript = asr.transcribe_samples(&samples, ASR_MAX_TOKENS)?;
            let asr_secs = t.elapsed().as_secs_f64();
            if transcript.trim().is_empty() {
                eprintln!("turn {turn}: empty transcript, skipping (asr {asr_secs:.2}s)");
                speaking.store(false, Ordering::Relaxed);
                continue;
            }
            println!("you: {transcript}");
            let _ = std::io::stdout().flush();

            // LLM streamed; sentences synthesized as they complete. The sink
            // and synth_and_play both check `barge` so a new interrupt aborts
            // the reply mid-stream.
            let t_llm = Instant::now();
            let mut splitter = SentenceSplitter::new();
            let mut first_sentence_at: Option<f64> = None;
            let mut n_sentences = 0usize;
            let mut sink = |delta: &str| {
                if barge.load(Ordering::Relaxed) {
                    return; // barge-in: stop producing new sentences
                }
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
                        Some(&barge),
                        ref_codes.as_ref(),
                    );
                }
            };
            let llm_stats = match chat::generate_reply_with_system(
                &mut llm,
                &tokenizer,
                &device,
                Some(
                    "你是一个语音助手的回复模块。直接用一两句简短的口语回答用户，禁止输出思考过程、分析、复述用户问题或任何 markdown 格式。",
                ),
                &transcript,
                true,
                max_tokens,
                &mut sink,
                Some(&barge),
            ) {
                Ok(stats) => stats,
                Err(e) => {
                    eprintln!("turn {turn}: llm error: {e}; skipping turn");
                    speaking.store(false, Ordering::Relaxed);
                    continue;
                }
            };
            drop(sink);
            if !barge.load(Ordering::Relaxed)
                && let Some(rest) = splitter.flush()
            {
                if first_sentence_at.is_none() {
                    first_sentence_at = Some(t_llm.elapsed().as_secs_f64());
                }
                n_sentences += 1;
                println!("tiny: {rest}");
                let _ = std::io::stdout().flush();
                synth_and_play(
                    &mut tts,
                    &speaker,
                    &mut sim_out,
                    tts_sr,
                    n_sentences,
                    &rest,
                    Some(&barge),
                    ref_codes.as_ref(),
                );
            }
            if n_sentences == 0 {
                eprintln!("turn {turn}: empty reply, skipping");
                speaking.store(false, Ordering::Relaxed);
                continue;
            }
            eprintln!(
                "turn {turn} timings: asr {asr_secs:.2}s, llm first sentence {:.2}s ({} tokens, {:.1} tok/s), {n_sentences} sentence(s)",
                first_sentence_at.unwrap_or_default(),
                llm_stats.tokens,
                llm_stats.tokens as f64 / llm_stats.decode.as_secs_f64().max(1e-9),
            );
            if barge.load(Ordering::Relaxed) {
                speaker_arc.clear();
            }
            speaking.store(false, Ordering::Relaxed);
        }
        return Ok(());
    }
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
        let frame: Vec<f32> = if mic_gain_val != 1.0 {
            frame.iter().map(|s| (s * mic_gain_val).clamp(-1.0, 1.0)).collect()
        } else {
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
        if samples.len() < min_segment_samples(min_segment_samples_cli) {
            eprintln!(
                "turn skipped: segment too short ({:.2}s)",
                samples.len() as f32 / VAD_SAMPLE_RATE as f32
            );
            continue;
        }
        // Energy gate: drop distant/bystander speech.
        let peak = samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, |a, b| a.max(b));
        let gate = min_seg_peak(min_seg_peak_cli);
        if peak < gate {
            eprintln!(
                "turn skipped: peak {peak:.3} < gate {gate:.3} (distant/noise)"
            );
            continue;
        }
        eprintln!("segment peak {peak:.3}");
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
                    None,
                    ref_codes.as_ref(),
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
            None,
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
            synth_and_play(
                &mut tts,
                &speaker,
                &mut sim_out,
                tts_sr,
                n_sentences,
                &rest,
                None,
                ref_codes.as_ref(),
            );
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
