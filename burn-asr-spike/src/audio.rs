//! CPU-only audio pipeline: decode (symphonia) → mono → sinc resample to 16kHz
//! → whisper-style mel (STFT via realfft + slaney mel filter bank).
//!
//! Ported 1:1 from tiny-cpm src/utils/audio_utils.rs + src/models/feature_extractor/
//! feature_extraction_whisper.rs. The candle version runs parts of this on GPU;
//! here the whole pipeline is plain f32 on the CPU (the mel frontend is a tiny
//! fraction of the transformer cost, so this does not affect the RTF comparison).

use std::sync::Arc;

use anyhow::{Result, anyhow};
use realfft::{RealFftPlanner, RealToComplex};
use symphonia::core::{
    audio::{AudioBufferRef, SampleBuffer},
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
};

pub const TARGET_SR: usize = 16000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const MEL_BINS: usize = 128;
const MAX_ASR_INPUT_SECONDS: f32 = 1200.0;

/// Decode an audio file to mono f32 samples + sample rate.
pub fn decode_audio(path: &str) -> Result<(Vec<f32>, usize)> {
    let bytes = std::fs::read(path)?;
    decode_bytes(&bytes)
}

fn decode_bytes(bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
    let extension = get_audio_format_from_bytes(bytes)?;
    let mss = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes.to_vec())),
        MediaSourceStreamOptions::default(),
    );
    let mut hint = Hint::new();
    hint.with_extension(&extension);
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no default track"))?;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0) as usize;
    if sample_rate == 0 {
        return Err(anyhow!("unknown sample rate"));
    }
    if track.codec_params.codec == CODEC_TYPE_NULL {
        return Err(anyhow!("null codec"));
    }
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut channels = 0usize;
    let mut all: Vec<Vec<f32>> = Vec::new();
    while let Ok(packet) = format.next_packet() {
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec().clone();
                channels = spec.channels.count();
                // per-channel contiguous samples, exactly like candle's
                // load_audio_use_symphonia (buf.chan(ch)); slicing the
                // interleaved buffer into contiguous blocks mixes L/R together.
                let mut buf = SampleBuffer::<f32>::new(decoded.frames() as u64, spec);
                buf.copy_interleaved_ref(decoded);
                let interleaved = buf.samples().to_vec();
                let per = interleaved.len() / channels;
                for ch in 0..channels {
                    if all.len() <= ch {
                        all.push(Vec::new());
                    }
                    all[ch].extend(
                        interleaved
                            .iter()
                            .skip(ch)
                            .step_by(channels)
                            .take(per)
                            .copied(),
                    );
                }
            }
            Err(e) => {
                eprintln!("decode error: {e}");
                break;
            }
        }
    }
    if all.is_empty() {
        return Err(anyhow!("no audio samples decoded"));
    }
    // mono: mean over channels (matches candle's mean_keepdim(0) for >1ch)
    let mono: Vec<f32> = if all.len() == 1 {
        all.into_iter().next().unwrap()
    } else {
        let n = all[0].len();
        (0..n)
            .map(|i| all.iter().map(|c| c[i]).sum::<f32>() / all.len() as f32)
            .collect()
    };
    Ok((mono, sample_rate))
}

fn get_audio_format_from_bytes(bytes: &[u8]) -> Result<&'static str> {
    if bytes.len() < 12 {
        return Err(anyhow!("bytes too short"));
    }
    if bytes.starts_with(&[0x52, 0x49, 0x46, 0x46]) && bytes.len() >= 12 {
        return if bytes[8..12] == [0x57, 0x41, 0x56, 0x45] {
            Ok("wav")
        } else {
            Ok("riff")
        };
    }
    if bytes.starts_with(&[0xFF, 0xFB]) || bytes.starts_with(&[0xFF, 0xF3]) || bytes.starts_with(&[0xFF, 0xF2]) {
        return Ok("mp3");
    }
    if bytes.len() >= 3 && bytes[0..3] == [0x49, 0x44, 0x33] {
        return Ok("mp3");
    }
    if bytes.len() >= 4 && bytes[0..4] == [0x46, 0x4F, 0x52, 0x4D] {
        return Ok("aiff");
    }
    if bytes.len() >= 8 && bytes[0..4] == [0x4F, 0x67, 0x67, 0x53] {
        return Ok("ogg");
    }
    if bytes.len() >= 4 && bytes[0..4] == [0x66, 0x4C, 0x61, 0x43] {
        return Ok("flac");
    }
    Err(anyhow!("unknown audio format"))
}

fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Port of torchaudio's sinc resample (lowpass_filter_width=6, rolloff=0.99,
/// Hann window), as implemented by tiny-cpm's get_sinc_resample_kernel +
/// apply_sinc_resample_kernel. Pure f32 CPU version; the candle one runs a
/// Conv1d on the GPU but the math is identical.
pub fn resample_sinc(waveform: &[f32], orig_freq: usize, new_freq: usize) -> Vec<f32> {
    let gcd_val = gcd(orig_freq, new_freq);
    let of = (orig_freq / gcd_val) as i64;
    let nf = (new_freq / gcd_val) as i64;
    if of == nf {
        return waveform.to_vec();
    }
    let lowpass_filter_width = 6i64;
    let rolloff = 0.99f64;
    let base_freq = (of.min(nf) as f64) * rolloff;
    let width_f = lowpass_filter_width as f64 * of as f64 / base_freq;
    let width = width_f.ceil() as i64;
    let klen = (of + 2 * width) as usize;

    // kernel[out][i]: t = (i - width)/of + (out - (nf-1))/nf, times base_freq
    let mut kernel = vec![0f32; (nf * klen as i64) as usize];
    for out in 0..nf {
        for i in 0..klen as i64 {
            let t = ((i - width) as f64 / of as f64 + (out - (nf - 1)) as f64 / nf as f64)
                * base_freq;
            let t = t.clamp(-lowpass_filter_width as f64, lowpass_filter_width as f64);
            let window_arg = t * std::f64::consts::PI / (lowpass_filter_width as f64) / 2.0;
            let window = window_arg.cos().powi(2);
            let sinc = if t == 0.0 {
                1.0
            } else {
                (t * std::f64::consts::PI).sin() / (t * std::f64::consts::PI)
            };
            let scale = base_freq / of as f64;
            kernel[(out * klen as i64 + i) as usize] = (sinc * window * scale) as f32;
        }
    }

    // pad left `width`, right `width + of` zeros, then strided conv
    let len = waveform.len();
    let mut padded = vec![0f32; len + (width + width + of) as usize];
    padded[width as usize..width as usize + len].copy_from_slice(waveform);
    let target_len = ((new_freq as f64 * len as f64) / orig_freq as f64).ceil() as usize;
    let mut out = vec![0f32; target_len];
    for p in 0..target_len {
        let base = p * of as usize;
        for o in 0..nf as usize {
            let mut acc = 0f64;
            let krow = o * klen;
            for i in 0..klen {
                acc += kernel[krow + i] as f64 * padded[base + i] as f64;
            }
            let idx = p * nf as usize + o;
            if idx < target_len {
                out[idx] = acc as f32;
            }
        }
    }
    out
}

/// Peak-normalize to [-1, 1] (candle float_range_normalize).
fn float_range_normalize(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if peak == 0.0 {
        return;
    }
    if peak > 1.0 {
        let inv = 1.0 / peak;
        for x in samples.iter_mut() {
            *x *= inv;
        }
    }
    for x in samples.iter_mut() {
        *x = x.clamp(-1.0, 1.0);
    }
}

fn reflect_pad_last_dim(samples: &[f32], pad_l: usize, pad_r: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + pad_l + pad_r);
    // left: reversed samples[1..=pad_l] (candle narrow(1, 1, pad_l).flip)
    for i in (1..=pad_l).rev() {
        out.push(samples[i.min(n - 1)]);
    }
    out.extend_from_slice(samples);
    // right: reversed samples[n-pad_r..n]
    for i in (n.saturating_sub(pad_r)..n).rev() {
        out.push(samples[i]);
    }
    out
}

fn hann_window(n: usize) -> Vec<f32> {
    // candle create_hann_window: i in (1-n)..n step 2, 0.5 + 0.5*cos(pi*i/(n-1))
    let denom = (n - 1) as f64;
    let start = 1i64 - n as i64;
    let end = n as i64;
    (start..end)
        .step_by(2)
        .map(|i| (0.5 + 0.5 * (std::f64::consts::PI * i as f64 / denom).cos()) as f32)
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum MelScale {
    Slaney,
}

fn hertz_to_mel(freq: f32, _scale: MelScale) -> f32 {
    // Slaney
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4_f32.ln();
    let mut mels = 3.0 * freq / 200.0;
    if freq >= min_log_hertz {
        mels = min_log_mel + (freq / min_log_hertz).ln() * logstep;
    }
    mels
}

fn mel_to_hertz(mels: f32, _scale: MelScale) -> f32 {
    // Slaney
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4_f32.ln() / 27.0;
    let mut freq = 200.0 * mels / 3.0;
    if mels >= min_log_mel {
        freq = min_log_hertz * (logstep * (mels - min_log_mel)).exp();
    }
    freq
}

fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    (0..steps).map(|i| start + i as f32 * step).collect()
}

/// Slaney-normalized mel filter bank, shape (num_mel, num_bins) row-major.
/// Port of candle mel_filter_bank(.., "slaney") with triangularize_in_mel_space=false.
fn mel_filter_bank(num_bins: usize, num_mel: usize, sr: f32) -> Vec<f32> {
    let mel_min = hertz_to_mel(0.0, MelScale::Slaney);
    let mel_max = hertz_to_mel(sr / 2.0, MelScale::Slaney);
    let mel_freqs = linspace_f32(mel_min, mel_max, num_mel + 2);
    let filter_freqs: Vec<f32> = mel_freqs.iter().map(|&m| mel_to_hertz(m, MelScale::Slaney)).collect();
    let fft_freqs = linspace_f32(0.0, sr / 2.0, num_bins);

    // triangular filters (num_bins, num_mel)
    let mut filters = vec![0f32; num_bins * num_mel];
    for j in 0..num_bins {
        for i in 0..num_mel {
            let down = -(filter_freqs[i] - fft_freqs[j]) / (filter_freqs[i + 1] - filter_freqs[i]);
            let up = (filter_freqs[i + 2] - fft_freqs[j]) / (filter_freqs[i + 2] - filter_freqs[i + 1]);
            let v = down.min(up).max(0.0);
            // slaney normalization
            let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
            filters[j * num_mel + i] = v * enorm;
        }
    }
    // transpose to (num_mel, num_bins)
    let mut out = vec![0f32; num_mel * num_bins];
    for i in 0..num_mel {
        for j in 0..num_bins {
            out[i * num_bins + j] = filters[j * num_mel + i];
        }
    }
    out
}

struct Stft {
    plan: Arc<dyn RealToComplex<f32>>,
}

impl Stft {
    fn new(n_fft: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(n_fft);
        Stft { plan }
    }
    /// power spectrum of one frame (n_fft/2 + 1 bins)
    fn power(&mut self, frame: &mut [f32]) -> Vec<f32> {
        let mut spectrum = self.plan.make_output_vec();
        self.plan.process(frame, &mut spectrum).unwrap();
        spectrum.iter().map(|c| c.norm_sqr()).collect()
    }
}

/// Full mel frontend. Returns (mel, n_mel_frames) where mel is row-major
/// (128, n_frames - 1) and n_mel_frames = n_frames - 1.
/// Mirrors WhisperFeatureExtractor::extract_fbank_features.
pub fn compute_mel(samples_16k: &[f32]) -> Result<(Vec<f32>, usize)> {
    let mut wav = samples_16k.to_vec();
    float_range_normalize(&mut wav);

    let pad = N_FFT / 2; // 200
    let padded = reflect_pad_last_dim(&wav, pad, pad);
    let window = hann_window(N_FFT);
    let filters = mel_filter_bank(1 + N_FFT / 2, MEL_BINS, TARGET_SR as f32);

    let n_frames = 1 + (padded.len() - N_FFT) / HOP;
    let n_use = n_frames - 1;
    let mut stft = Stft::new(N_FFT);
    let mut log_spec = vec![0f32; MEL_BINS * n_use];

    for t in 0..n_use {
        let start = t * HOP;
        // frame * window (hann) in the time domain, then FFT (candle torch_stft)
        let mut frame: Vec<f32> = padded[start..start + N_FFT]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();
        let power = stft.power(&mut frame);
        // mel: (128, 201) @ (201,) → (128,)
        for m in 0..MEL_BINS {
            let mut acc = 0.0f64;
            let frow = m * (1 + N_FFT / 2);
            for (j, f) in power.iter().enumerate() {
                acc += filters[frow + j] as f64 * *f as f64;
            }
            let v = (acc as f32).clamp(1e-10, f32::INFINITY);
            let v = v.log10();
            log_spec[m * n_use + t] = v;
        }
    }

    // global max - 8 floor, then (x + 4) / 4
    let max_val = log_spec.iter().cloned().fold(f32::MIN, f32::max) - 8.0;
    for v in log_spec.iter_mut() {
        *v = (*v).max(max_val);
        *v = (*v + 4.0) / 4.0;
    }
    Ok((log_spec, n_use))
}

/// tiny-cpm get_feat_extract_output_lengths: 100-frame window → 13 frames
/// (3 stride-2 convs), remainder through the conv chain alone.
pub fn get_feat_extract_output_lengths(audio_len: usize) -> usize {
    let input_len_leave = audio_len % 100;
    if input_len_leave > 0 {
        let feat_lengths = (input_len_leave - 1) / 2 + 1;
        ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1 + (audio_len / 100) * 13
    } else {
        (audio_len / 100) * 13
    }
}

/// Split a long waveform into <= max_chunk_sec chunks (candle split_audio_into_chunks).
pub fn split_into_chunks(samples: &[f32]) -> Vec<Vec<f32>> {
    let total_sec = samples.len() as f32 / TARGET_SR as f32;
    if total_sec <= MAX_ASR_INPUT_SECONDS {
        return vec![samples.to_vec()];
    }
    let max_len = (MAX_ASR_INPUT_SECONDS * TARGET_SR as f32).round() as usize;
    let split_len = samples.len() / max_len;
    let mut chunks = Vec::new();
    for i in 0..split_len {
        chunks.push(samples[i * max_len..(i + 1) * max_len].to_vec());
    }
    let remain = samples.len() % max_len;
    if remain > 0 {
        chunks.push(samples[split_len * max_len..].to_vec());
    }
    chunks
}
