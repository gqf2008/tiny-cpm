//! Live audio IO for the `live` subcommand, via cpal (CoreAudio on macOS).
//!
//! - [`MicCapture`]: default input device -> 16kHz mono f32 -> 400-sample
//!   frames over an mpsc channel (ready for `FireRedVad::detect_frame_f32`).
//! - [`Speaker`]: default output device; the cpal callback drains a shared
//!   f32 queue (already at device rate/channels), emitting silence when empty.
//!
//! Resampling reuses `audio_utils::resample_simple` (sinc, CPU tensors). It
//! runs on worker/caller threads, never inside the realtime cpal callbacks.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, mpsc};

use anyhow::{Result, anyhow, bail};
use candle_core::{Device, Tensor};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::utils::audio_utils::resample_simple;

/// VAD/ASR sample rate.
pub const VAD_SAMPLE_RATE: usize = 16000;
/// VAD frame size at 16kHz (25 ms).
pub const VAD_FRAME_SAMPLES: usize = 400;

/// Microphone capture on the default input device, delivered as 400-sample
/// 16kHz mono f32 frames.
pub struct MicCapture {
    rx: mpsc::Receiver<Vec<f32>>,
    // Dropping the stream stops capture; keep it alive for struct lifetime.
    _stream: cpal::Stream,
}

impl MicCapture {
    /// Open the default input device and start capturing. A worker thread
    /// resamples device-rate chunks to 16kHz and slices 400-sample frames.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default audio input device"))?;
        let config = device
            .default_input_config()
            .map_err(|e| anyhow!("default_input_config: {e}"))?;
        let device_sr = config.sample_rate().0 as usize;
        let device_channels = config.channels() as usize;
        eprintln!(
            "mic: {} Hz, {} ch, {:?}",
            device_sr,
            device_channels,
            config.sample_format()
        );

        // cpal callback -> raw mono f32 chunks (device rate) -> worker thread.
        let (tx_raw, rx_raw) = mpsc::channel::<Vec<f32>>();
        let err_fn = |e| eprintln!("mic stream error: {e}");
        let stream_config: cpal::StreamConfig = config.clone().into();
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    let _ = tx_raw.send(mix_to_mono(data, device_channels));
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let f: Vec<f32> = data.iter().map(|s| *s as f32 / 32768.0).collect();
                    let _ = tx_raw.send(mix_to_mono(&f, device_channels));
                },
                err_fn,
                None,
            ),
            other => bail!("unsupported mic sample format: {other:?}"),
        }
        .map_err(|e| anyhow!("build_input_stream: {e}"))?;
        stream.play().map_err(|e| anyhow!("mic play: {e}"))?;

        // Worker: resample chunks to 16kHz, emit exact 400-sample frames.
        let (tx_frames, rx_frames) = mpsc::channel::<Vec<f32>>();
        std::thread::spawn(move || {
            let cpu = Device::Cpu;
            let mut pending: Vec<f32> = Vec::new();
            while let Ok(chunk) = rx_raw.recv() {
                let chunk = match resample_to_16k(&chunk, device_sr, &cpu) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("mic resample error: {e}");
                        continue;
                    }
                };
                pending.extend_from_slice(&chunk);
                while pending.len() >= VAD_FRAME_SAMPLES {
                    let frame: Vec<f32> = pending.drain(..VAD_FRAME_SAMPLES).collect();
                    if tx_frames.send(frame).is_err() {
                        return; // receiver dropped
                    }
                }
            }
        });

        Ok(Self {
            rx: rx_frames,
            _stream: stream,
        })
    }

    /// Block until the next 400-sample 16kHz mono frame arrives.
    pub fn next_frame(&self) -> Result<Vec<f32>> {
        self.rx
            .recv()
            .map_err(|_| anyhow!("mic capture stream closed"))
    }
}

/// Speaker playback on the default output device. Samples pushed via
/// [`Speaker::push`] are converted to the device rate/channel layout and
/// appended to a shared queue drained by the cpal callback.
pub struct Speaker {
    queue: Arc<Mutex<VecDeque<f32>>>,
    device_sr: usize,
    device_channels: usize,
    _stream: cpal::Stream,
}

impl Speaker {
    /// Open the default output device and start the playback stream.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default audio output device"))?;
        let config = device
            .default_output_config()
            .map_err(|e| anyhow!("default_output_config: {e}"))?;
        let device_sr = config.sample_rate().0 as usize;
        let device_channels = config.channels() as usize;
        eprintln!(
            "speaker: {} Hz, {} ch, {:?}",
            device_sr,
            device_channels,
            config.sample_format()
        );

        let queue: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
        let pop = |queue: &Arc<Mutex<VecDeque<f32>>>, out: &mut [f32]| {
            // try_lock: never block the realtime callback (and never panic on
            // a poisoned mutex). Contention/underrun -> silence.
            if let Ok(mut q) = queue.try_lock() {
                for sample in out.iter_mut() {
                    *sample = q.pop_front().unwrap_or(0.0);
                }
            } else {
                out.fill(0.0);
            }
        };
        let err_fn = |e| eprintln!("speaker stream error: {e}");
        let stream_config: cpal::StreamConfig = config.clone().into();
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let q = queue.clone();
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [f32], _| pop(&q, data),
                    err_fn,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let q = queue.clone();
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [i16], _| {
                        let mut tmp = vec![0.0f32; data.len()];
                        pop(&q, &mut tmp);
                        for (dst, src) in data.iter_mut().zip(tmp) {
                            *dst = (src.clamp(-1.0, 1.0) * 32767.0) as i16;
                        }
                    },
                    err_fn,
                    None,
                )
            }
            other => bail!("unsupported speaker sample format: {other:?}"),
        }
        .map_err(|e| anyhow!("build_output_stream: {e}"))?;
        stream.play().map_err(|e| anyhow!("speaker play: {e}"))?;

        Ok(Self {
            queue,
            device_sr,
            device_channels,
            _stream: stream,
        })
    }

    /// Queue interleaved f32 PCM (src_channels, src_sr) for playback.
    /// Fast path: layout already matches the device. Otherwise the audio is
    /// mixed to mono, resampled, and duplicated across the device channels.
    /// Pushes are chunked into ~100 ms slices so the output callback's
    /// try_lock is never starved by one bulk enqueue.
    pub fn push(&self, pcm: &[f32], src_sr: usize, src_channels: usize) -> Result<()> {
        if src_sr == self.device_sr && src_channels == self.device_channels {
            let slice = (self.device_sr * self.device_channels / 10).max(1);
            for chunk in pcm.chunks(slice) {
                self.queue.lock().unwrap().extend(chunk.iter().copied());
            }
            return Ok(());
        }
        // Deinterleave -> mono.
        let frames: Vec<f32> = pcm
            .chunks_exact(src_channels)
            .map(|f| f.iter().sum::<f32>() / src_channels as f32)
            .collect();
        let mono = resample_to(&frames, src_sr, self.device_sr, &Device::Cpu)?;
        let mut expanded = Vec::with_capacity(mono.len() * self.device_channels);
        for s in mono {
            for _ in 0..self.device_channels {
                expanded.push(s);
            }
        }
        let slice = (self.device_sr * self.device_channels / 10).max(1);
        for chunk in expanded.chunks(slice) {
            self.queue.lock().unwrap().extend(chunk.iter().copied());
        }
        Ok(())
    }

    /// Seconds of audio currently queued for playback.
    pub fn queued_seconds(&self) -> f32 {
        let q = self.queue.lock().unwrap();
        q.len() as f32 / (self.device_sr * self.device_channels) as f32
    }

    /// Drop all queued audio (failsafe for a stuck/dead output stream).
    pub fn clear(&self) {
        self.queue.lock().unwrap().clear();
    }

    /// A clone of the shared playback queue handle. The queue is
    /// `Arc<Mutex<..>>` (Send) so it can be handed to another thread that
    /// needs to clear the queue (e.g. barge-in listener) without moving the
    /// !Send `Speaker` (which owns the cpal playback stream).
    pub fn shared_queue(&self) -> Arc<Mutex<VecDeque<f32>>> {
        Arc::clone(&self.queue)
    }
}

/// Interleaved multi-channel -> mono (average).
fn mix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    data.chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample a mono f32 chunk to 16kHz (no-op if already there).
fn resample_to_16k(samples: &[f32], orig_sr: usize, device: &Device) -> Result<Vec<f32>> {
    resample_to(samples, orig_sr, VAD_SAMPLE_RATE, device)
}

fn resample_to(
    samples: &[f32],
    orig_sr: usize,
    target_sr: usize,
    device: &Device,
) -> Result<Vec<f32>> {
    if orig_sr == target_sr {
        return Ok(samples.to_vec());
    }
    let t = Tensor::new(samples, device)?.unsqueeze(0)?;
    let t = resample_simple(&t, orig_sr as i64, target_sr as i64)?;
    Ok(t.squeeze(0)?.to_vec1::<f32>()?)
}

#[cfg(test)]
mod tests {
    use super::resample_to;

    /// 48kHz -> 16kHz must preserve duration (len/3) and frequency content.
    #[test]
    fn resample_48k_to_16k_preserves_rate() {
        let n = 48000; // 1s of 440Hz sine at 48kHz
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        let out = resample_to(&samples, 48000, 16000, &candle_core::Device::Cpu).unwrap();
        assert!(
            (out.len() as f32 - 16000.0).abs() < 50.0,
            "expected ~16000 samples, got {}",
            out.len()
        );
        let crossings = out.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            (crossings as f32 - 880.0).abs() < 60.0,
            "expected ~880 zero crossings, got {crossings}"
        );
    }
}
