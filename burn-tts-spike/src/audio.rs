//! CPU-only audio pipeline: decode (symphonia) → mono → sinc resample → the
//! speaker-encoder mel frontend (STFT via realfft + slaney mel filter bank).
//! Ported 1:1 from burn-asr-spike's verified audio.rs + tiny-cpm's
//! speaker_encoder.rs `mel_spectrogram` (reflect-pad 384, periodic Hann
//! 1024/256, magnitude sqrt(power+1e-9), 128-bin slaney mel @ 24 kHz,
//! log(clamp 1e-5)). Pure f32 on the CPU, like the candle reference.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use realfft::{RealFftPlanner, RealToComplex};
use symphonia::core::{
    audio::{AudioBufferRef, SampleBuffer},
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
};

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
    if bytes.starts_with(&[0xFF, 0xFB])
        || bytes.starts_with(&[0xFF, 0xF3])
        || bytes.starts_with(&[0xFF, 0xF2])
    {
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
            let t =
                ((i - width) as f64 / of as f64 + (out - (nf - 1)) as f64 / nf as f64) * base_freq;
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

/// Periodic Hann — torch hann_window(periodic=True), same as candle's
/// cosyvoice3 s3tok periodic_hann (which the speaker encoder uses; NOT the
/// create_hann_window formula the ASR frontend uses).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

fn hertz_to_mel(freq: f32) -> f32 {
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

fn mel_to_hertz(mels: f32) -> f32 {
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

/// Slaney-normalized mel filter bank, (num_mel, num_bins) row-major.
/// fmin=0, fmax=sr/2 — exactly the speaker encoder's params (fmax 12000 @ 24k).
fn mel_filter_bank(num_bins: usize, num_mel: usize, sr: f32) -> Vec<f32> {
    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(sr / 2.0);
    let mel_freqs = linspace_f32(mel_min, mel_max, num_mel + 2);
    let filter_freqs: Vec<f32> = mel_freqs.iter().map(|&m| mel_to_hertz(m)).collect();
    let fft_freqs = linspace_f32(0.0, sr / 2.0, num_bins);

    // triangular filters (num_bins, num_mel)
    let mut filters = vec![0f32; num_bins * num_mel];
    for j in 0..num_bins {
        for i in 0..num_mel {
            let down = -(filter_freqs[i] - fft_freqs[j]) / (filter_freqs[i + 1] - filter_freqs[i]);
            let up =
                (filter_freqs[i + 2] - fft_freqs[j]) / (filter_freqs[i + 2] - filter_freqs[i + 1]);
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

/// Reflect-pad both sides (candle pad_reflect_last_dim: left reverses
/// samples[1..=pad_l], right reverses the last pad_r samples).
fn reflect_pad_last_dim(samples: &[f32], pad_l: usize, pad_r: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + pad_l + pad_r);
    for i in (1..=pad_l).rev() {
        out.push(samples[i.min(n - 1)]);
    }
    out.extend_from_slice(samples);
    for i in (n.saturating_sub(pad_r)..n).rev() {
        out.push(samples[i]);
    }
    out
}

/// Speaker-encoder mel frontend: wav @ 24 kHz → row-major (T, mel_dim) log-mel.
/// Mirrors candle's `SpeakerEncoder::mel_spectrogram`: if the input is shorter
/// than the reflect pad, extend with one zero first; no frame dropping.
pub fn speaker_mel(samples_24k: &[f32], n_fft: usize, hop: usize, mel_dim: usize) -> Vec<f32> {
    let pad = (n_fft - hop) / 2; // 384
    let mut pcm = samples_24k.to_vec();
    if pcm.len() <= pad {
        pcm.resize(pad + 1, 0.0);
    }
    let padded = reflect_pad_last_dim(&pcm, pad, pad);
    let window = hann_window(n_fft);
    let filters = mel_filter_bank(1 + n_fft / 2, mel_dim, 24000.0);
    let n_frames = 1 + (padded.len() - n_fft) / hop;
    let mut stft = Stft::new(n_fft);
    let mut out = vec![0f32; n_frames * mel_dim];
    for t in 0..n_frames {
        let start = t * hop;
        let mut frame: Vec<f32> = padded[start..start + n_fft]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();
        let power = stft.power(&mut frame); // (n_fft/2+1,)
        let n_bins = 1 + n_fft / 2;
        for m in 0..mel_dim {
            let mut acc = 0.0f64;
            let frow = m * n_bins;
            for (j, p) in power.iter().enumerate() {
                acc += filters[frow + j] as f64 * (*p as f64 + 1e-9).sqrt();
            }
            // mel = mag @ filters; log(clamp 1e-5).
            let v = (acc as f32).clamp(1e-5, f32::INFINITY).ln();
            out[t * mel_dim + m] = v;
        }
    }
    out
}
