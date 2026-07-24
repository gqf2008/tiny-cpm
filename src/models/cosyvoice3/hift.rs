//! CosyVoice3 CausalHiFTGenerator vocoder (mel -> 24 kHz waveform), ported from
//! CrispASR's C++/ggml implementation (src/cosyvoice3_tts.cpp, phase 4).
//!
//! Pipeline: mel -> F0 predictor (causal conv net) -> SineGen2 harmonic source
//! -> STFT -> HiFi-GAN-style causal upsample tower with source fusion ->
//! conv_post -> half-spectrum iSTFT (n_fft=16, hop=4).
//!
//! All convs upstream are weight-normed (`weight_g`/`weight_v` parametrized);
//! the loader materializes `w = g * v / ||v||` exactly like the GGUF converter
//! (convert-cosyvoice3-to-gguf.py `wn_resolve`). The conv tower runs on the
//! target device (Metal); SineGen/STFT/iSTFT run CPU-side on plain Vec<f32>
//! (tiny n_fft=16, naive O(n^2) DFT), mirroring the C++ split.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, pickle::read_all_with_key};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

use crate::utils::tensor_utils::repeat_interleave;

const SAMPLE_RATE: usize = 24000;
const MEL_DIM: usize = 80;
const BASE_CHANNELS: usize = 512;
const NB_HARMONICS: usize = 9; // 1 fundamental + 8 harmonics
const UPSAMPLE_SCALE: usize = 480; // mel frame -> audio samples
const ISTFT_N_FFT: usize = 16;
const ISTFT_HOP: usize = 4;
const N_FREQ: usize = ISTFT_N_FFT / 2 + 1; // 9
const S_STFT_CH: usize = ISTFT_N_FFT + 2; // 18 (real ++ imag)
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
const DOWN_STRIDES: [usize; 3] = [15, 3, 1];
const DILATIONS: [usize; 3] = [1, 3, 5];
const LRELU_SLOPE: f64 = 0.1;
const AUDIO_LIMIT: f32 = 0.99;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THR: f32 = 10.0;
/// Fixed seed for the SineGen uniform[0,1) noise buffer — upstream draws it
/// from torch's global RNG; we just need deterministic noise.
const NOISE_SEED: u64 = 22222;

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

fn take(map: &mut HashMap<String, Tensor>, key: &str) -> Result<Tensor> {
    map.remove(key)
        .ok_or_else(|| anyhow!("hift.pt missing tensor `{key}`"))?
        .to_dtype(DType::F32)
        .map_err(|e| anyhow!("hift tensor `{key}` to f32: {e}"))
}

/// Materialize PyTorch weight_norm (dim=0) into a plain weight:
/// w = g * v / max(||v||, 1e-12), per output channel.
fn resolve_wn(map: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Tensor> {
    let g = take(map, &format!("{prefix}.parametrizations.weight.original0"))?;
    let v = take(map, &format!("{prefix}.parametrizations.weight.original1"))?;
    let mut norm = v.sqr()?;
    for d in 1..v.rank() {
        norm = norm.sum_keepdim(d)?;
    }
    let norm = norm.sqrt()?.clamp(1e-12f64, f64::MAX)?;
    Ok(v.broadcast_div(&norm)?.broadcast_mul(&g)?)
}

struct HiftResBlock {
    c1: [(Tensor, Tensor); 3], // (weight, bias), dilations [1, 3, 5]
    c2: [(Tensor, Tensor); 3], // dilation 1
    a1: [Tensor; 3],           // Snake alpha, (1, C, 1)
    a2: [Tensor; 3],
}

fn load_resblock(
    map: &mut HashMap<String, Tensor>,
    prefix: &str,
    device: &Device,
) -> Result<HiftResBlock> {
    let mut c1: Vec<(Tensor, Tensor)> = Vec::with_capacity(3);
    let mut c2: Vec<(Tensor, Tensor)> = Vec::with_capacity(3);
    let mut a1: Vec<Tensor> = Vec::with_capacity(3);
    let mut a2: Vec<Tensor> = Vec::with_capacity(3);
    for j in 0..3 {
        c1.push((
            resolve_wn(map, &format!("{prefix}.convs1.{j}"))?,
            take(map, &format!("{prefix}.convs1.{j}.bias"))?,
        ));
        c2.push((
            resolve_wn(map, &format!("{prefix}.convs2.{j}"))?,
            take(map, &format!("{prefix}.convs2.{j}.bias"))?,
        ));
        a1.push(take(map, &format!("{prefix}.activations1.{j}.alpha"))?);
        a2.push(take(map, &format!("{prefix}.activations2.{j}.alpha"))?);
    }
    let to_dev = |t: Tensor| -> Result<Tensor> { Ok(t.to_device(device)?) };
    let alpha = |t: Tensor| -> Result<Tensor> {
        // Snake alpha is stored per channel (any layout) -> (1, C, 1).
        let c = t.elem_count();
        Ok(t.flatten_all()?.reshape((1, c, 1))?.to_device(device)?)
    };
    let pair =
        |(w, b): (Tensor, Tensor)| -> Result<(Tensor, Tensor)> { Ok((to_dev(w)?, to_dev(b)?)) };
    Ok(HiftResBlock {
        c1: [
            pair(c1.remove(0))?,
            pair(c1.remove(0))?,
            pair(c1.remove(0))?,
        ],
        c2: [
            pair(c2.remove(0))?,
            pair(c2.remove(0))?,
            pair(c2.remove(0))?,
        ],
        a1: [
            alpha(a1.remove(0))?,
            alpha(a1.remove(0))?,
            alpha(a1.remove(0))?,
        ],
        a2: [
            alpha(a2.remove(0))?,
            alpha(a2.remove(0))?,
            alpha(a2.remove(0))?,
        ],
    })
}

// ---------------------------------------------------------------------------
// Tensor helpers (all tensors (1, C, T) on device)
// ---------------------------------------------------------------------------

/// 1D conv with explicit asymmetric zero padding on the time dim.
/// Weights are plain PyTorch-layout (out_c, in_c, k).
#[allow(clippy::too_many_arguments)]
fn conv1d_padded(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    pad_l: usize,
    pad_r: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let x = if pad_l + pad_r > 0 {
        x.pad_with_zeros(2, pad_l, pad_r)?
    } else {
        x.clone()
    };
    let y = x.conv1d(w, 0, stride, dilation, 1)?;
    Ok(y.broadcast_add(&b.reshape((1, b.elem_count(), 1))?)?)
}

/// Causal (left-padded) conv, kernel/dilation read from the weight shape.
fn conv1d_causal(x: &Tensor, w: &Tensor, b: &Tensor, dilation: usize) -> Result<Tensor> {
    let k = w.dim(2)?;
    conv1d_padded(x, w, b, (k - 1) * dilation, 0, 1, dilation)
}

/// Lookahead (right-padded) conv: output at t reads x[t..t+k).
fn conv1d_lookahead(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let k = w.dim(2)?;
    conv1d_padded(x, w, b, 0, k - 1, 1, 1)
}

/// Snake-Beta (alpha_logscale=False): x + sin^2(a*x) / (a + 1e-9).
fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let s2 = x.broadcast_mul(alpha)?.sin()?.sqr()?;
    let a_safe = (alpha + 1e-9f64)?;
    Ok(x.broadcast_add(&s2.broadcast_div(&a_safe)?)?)
}

impl HiftResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for j in 0..3 {
            let (w1, b1) = &self.c1[j];
            let (w2, b2) = &self.c2[j];
            let xt = snake(&x, &self.a1[j])?;
            let xt = conv1d_causal(&xt, w1, b1, DILATIONS[j])?;
            let xt = snake(&xt, &self.a2[j])?;
            let xt = conv1d_causal(&xt, w2, b2, 1)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// HiFT generator
// ---------------------------------------------------------------------------

pub struct Hift {
    conv_pre: (Tensor, Tensor),
    conv_post: (Tensor, Tensor),
    ups: [(Tensor, Tensor); 3],
    resblocks: Vec<HiftResBlock>,        // 9 = 3 stages x 3 kernels
    source_downs: [(Tensor, Tensor); 3], // plain convs (no weight-norm)
    source_resblocks: Vec<HiftResBlock>, // 3
    f0_condnet: Vec<(Tensor, Tensor)>,   // 5
    f0_classifier: (Tensor, Tensor),
    l_linear_w: Vec<f32>, // 9
    l_linear_b: f32,
    device: Device,
}

impl Hift {
    /// Load `hift.pt` (torch pickle) from `dir`.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = dir.as_ref().join("hift.pt");
        let mut map: HashMap<String, Tensor> = match read_all_with_key(&path, None) {
            Ok(d) => d.into_iter().collect(),
            Err(_) => read_all_with_key(&path, Some("state_dict"))
                .map_err(|e| anyhow!("read {} failed: {e}", path.display()))?
                .into_iter()
                .collect(),
        };

        let dev = |t: Tensor| -> Result<Tensor> { Ok(t.to_device(device)?) };
        let pair = |map: &mut HashMap<String, Tensor>, prefix: &str| -> Result<(Tensor, Tensor)> {
            Ok((
                dev(resolve_wn(map, prefix)?)?,
                dev(take(map, &format!("{prefix}.bias"))?)?,
            ))
        };
        // Plain (non weight-normed) conv.
        let plain = |map: &mut HashMap<String, Tensor>, prefix: &str| -> Result<(Tensor, Tensor)> {
            Ok((
                dev(take(map, &format!("{prefix}.weight"))?)?,
                dev(take(map, &format!("{prefix}.bias"))?)?,
            ))
        };

        let conv_pre = pair(&mut map, "conv_pre")?;
        let conv_post = pair(&mut map, "conv_post")?;
        let ups = [
            pair(&mut map, "ups.0")?,
            pair(&mut map, "ups.1")?,
            pair(&mut map, "ups.2")?,
        ];
        let mut resblocks = Vec::with_capacity(9);
        for i in 0..9 {
            resblocks.push(load_resblock(&mut map, &format!("resblocks.{i}"), device)?);
        }
        let source_downs = [
            plain(&mut map, "source_downs.0")?,
            plain(&mut map, "source_downs.1")?,
            plain(&mut map, "source_downs.2")?,
        ];
        let mut source_resblocks = Vec::with_capacity(3);
        for i in 0..3 {
            source_resblocks.push(load_resblock(
                &mut map,
                &format!("source_resblocks.{i}"),
                device,
            )?);
        }
        // F0 predictor: conv layers live at condnet.{0,2,4,6,8}.
        let mut f0_condnet = Vec::with_capacity(5);
        for k in [0, 2, 4, 6, 8] {
            f0_condnet.push(pair(&mut map, &format!("f0_predictor.condnet.{k}"))?);
        }
        let f0_classifier = plain(&mut map, "f0_predictor.classifier")?;

        let l_linear_w = take(&mut map, "m_source.l_linear.weight")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if l_linear_w.len() != NB_HARMONICS {
            return Err(anyhow!(
                "m_source.l_linear.weight has {} elems, expected {NB_HARMONICS}",
                l_linear_w.len()
            ));
        }
        let l_linear_b = take(&mut map, "m_source.l_linear.bias")?
            .flatten_all()?
            .to_vec1::<f32>()?
            .first()
            .copied()
            .ok_or_else(|| anyhow!("m_source.l_linear.bias empty"))?;

        Ok(Self {
            conv_pre,
            conv_post,
            ups,
            resblocks,
            source_downs,
            source_resblocks,
            f0_condnet,
            f0_classifier,
            l_linear_w,
            l_linear_b,
            device: device.clone(),
        })
    }

    /// mel (T_mel, 80) -> 24 kHz f32 samples, exactly T_mel * 480 long.
    pub fn mel_to_waveform(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let t_mel = mel.dim(0)?;
        if mel.dim(1)? != MEL_DIM {
            return Err(anyhow!("mel must be (T_mel, {MEL_DIM})"));
        }
        let mel = mel.to_dtype(DType::F32)?.to_device(&self.device)?;
        let x = mel.t()?.unsqueeze(0)?; // (1, 80, T_mel)

        let f0 = self.f0_predict(&x)?; // Vec<f32>, T_mel
        let t_stft = t_mel * UPSAMPLE_SCALE / ISTFT_HOP + 1;
        let s_stft = self.source_stft(&f0)?; // (18, T_stft) row-major
        let s = Tensor::from_vec(s_stft, (1, S_STFT_CH, t_stft), &self.device)?;

        let spec = self.decode(&x, &s)?; // (1, 18, T_stft)
        let spec = spec.squeeze(0)?.to_vec2::<f32>()?;
        Ok(istft(&spec, t_mel))
    }

    /// CausalConvRNNF0Predictor: mel -> |Linear(ELU conv tower)|, (T_mel,) Hz.
    fn f0_predict(&self, x: &Tensor) -> Result<Vec<f32>> {
        // Layer 0: lookahead conv (k=4, right-pad 3), 80 -> 512.
        let (w, b) = &self.f0_condnet[0];
        let mut h = conv1d_lookahead(x, w, b)?.elu(1.0)?;
        // Layers 1-4: causal conv (k=3, left-pad 2), 512 -> 512.
        for (w, b) in &self.f0_condnet[1..] {
            h = conv1d_causal(&h, w, b, 1)?.elu(1.0)?;
        }
        // Linear(512 -> 1) on (T, 512), then abs.
        let h = h.squeeze(0)?.t()?; // (T, 512)
        let y = h
            .matmul(&self.f0_classifier.0.t()?)?
            .broadcast_add(&self.f0_classifier.1)?; // (T, 1)
        Ok(y.squeeze(1)?.abs()?.to_vec1::<f32>()?)
    }

    /// SineGen2 + m_source.l_linear + tanh + STFT, all CPU.
    /// f0_mel (T_mel,) -> s_stft (18, T_stft) row-major (f-outer, t-inner).
    fn source_stft(&self, f0_mel: &[f32]) -> Result<Vec<f32>> {
        let t_mel = f0_mel.len();
        let t_audio = t_mel * UPSAMPLE_SCALE;

        // Downsampled rad at mel rate: with the nearest x480 f0, the upstream
        // linear-interp downsample collapses to f0_mel[t] (see CrispASR note).
        // phase = cumsum(rad) * 2pi * 480 (the *upsample_scale folded in).
        let two_pi_us = 2.0f32 * std::f32::consts::PI * UPSAMPLE_SCALE as f32;
        let mut phase_down = vec![0.0f32; t_mel * NB_HARMONICS];
        for h in 0..NB_HARMONICS {
            let mut acc = 0.0f32;
            for t in 0..t_mel {
                let r0 = f0_mel[t] * (h + 1) as f32 / SAMPLE_RATE as f32;
                acc += r0 - r0.floor();
                phase_down[t * NB_HARMONICS + h] = acc * two_pi_us;
            }
        }

        // sin + uv mask + seeded uniform noise, then l_linear + tanh.
        let mut rng = StdRng::seed_from_u64(NOISE_SEED);
        let uniform = Uniform::new(0.0f32, 1.0f32)?; // [0, 1)
        let mut sine_merge = vec![0.0f32; t_audio];
        for t in 0..t_audio {
            let tm = t / UPSAMPLE_SCALE;
            let uv = if f0_mel[tm] > VOICED_THR { 1.0f32 } else { 0.0 };
            let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
            let mut sum = self.l_linear_b;
            for h in 0..NB_HARMONICS {
                let sine = phase_down[tm * NB_HARMONICS + h].sin() * SINE_AMP * uv;
                let wave = sine + noise_amp * uniform.sample(&mut rng);
                sum += wave * self.l_linear_w[h];
            }
            sine_merge[t] = sum.tanh();
        }

        // STFT: n_fft=16, hop=4, periodic Hann, center=True + reflect pad.
        let n_pad = ISTFT_N_FFT / 2;
        let t_stft = t_audio / ISTFT_HOP + 1;
        let win = periodic_hann(ISTFT_N_FFT);
        let reflect_at = |idx: usize| -> f32 {
            let src = idx as i64 - n_pad as i64;
            if t_audio <= 1 {
                return if t_audio == 1 { sine_merge[0] } else { 0.0 };
            }
            let period = 2 * (t_audio as i64 - 1);
            let mut r = src.rem_euclid(period);
            if r >= t_audio as i64 {
                r = period - r;
            }
            sine_merge[r as usize]
        };
        let mut out = vec![0.0f32; S_STFT_CH * t_stft];
        for frame in 0..t_stft {
            let start = frame * ISTFT_HOP;
            for f in 0..N_FREQ {
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for n in 0..ISTFT_N_FFT {
                    let v = reflect_at(start + n) * win[n];
                    let angle = -2.0 * std::f32::consts::PI * (f * n) as f32 / ISTFT_N_FFT as f32;
                    re += v * angle.cos();
                    im += v * angle.sin();
                }
                out[f * t_stft + frame] = re;
                out[(N_FREQ + f) * t_stft + frame] = im;
            }
        }
        Ok(out)
    }

    /// Decode tower: (1, 80, T_mel) mel + (1, 18, T_stft) source STFT
    /// -> (1, 18, T_stft) log-mag/phase for the iSTFT.
    fn decode(&self, mel: &Tensor, s: &Tensor) -> Result<Tensor> {
        // conv_pre: lookahead conv (k=5, right-pad 4), 80 -> 512.
        let mut x = conv1d_lookahead(mel, &self.conv_pre.0, &self.conv_pre.1)?;

        for i in 0..3 {
            let ch_stage = BASE_CHANNELS >> (i + 1);
            // LeakyReLU(0.1) -> nearest upsample -> causal conv.
            x = candle_nn::ops::leaky_relu(&x, LRELU_SLOPE)?;
            x = repeat_interleave(&x, UPSAMPLE_RATES[i], 2)?;
            let (w, b) = &self.ups[i];
            x = conv1d_causal(&x, w, b, 1)?;
            debug_assert_eq!(x.dim(1)?, ch_stage);

            // TRAP: ReflectionPad1d((1, 0)) at the LAST stage only — the new
            // head sample is x[1] (reflect across the boundary, not x[0]).
            if i == 2 {
                let head = x.narrow(2, 1, 1)?;
                x = Tensor::cat(&[&head, &x], 2)?;
            }

            // Source fusion: strided causal downsample conv -> resblock -> add.
            let (w, b) = &self.source_downs[i];
            let stride = DOWN_STRIDES[i];
            let si = conv1d_padded(s, w, b, stride - 1, 0, stride, 1)?;
            let si = self.source_resblocks[i].forward(&si)?;
            // Defensive T alignment (dim math should already match).
            let t_min = x.dim(2)?.min(si.dim(2)?);
            let x_aligned = x.narrow(2, 0, t_min)?;
            let si = si.narrow(2, 0, t_min)?;
            x = (x_aligned + si)?;

            // Main fusion: 3 ResBlocks applied INDEPENDENTLY, outputs averaged.
            let mut xs: Option<Tensor> = None;
            for j in 0..3 {
                let rj = self.resblocks[i * 3 + j].forward(&x)?;
                xs = Some(match xs {
                    None => rj,
                    Some(acc) => (acc + rj)?,
                });
            }
            x = (xs.unwrap() * (1.0 / 3.0))?;
        }

        // TRAP: plain F.leaky_relu default slope 0.01 before conv_post.
        x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        conv1d_causal(&x, &self.conv_post.0, &self.conv_post.1, 1)
    }
}

fn periodic_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect()
}

/// Overlap-add iSTFT matching torch.istft(n_fft=16, hop=4, periodic Hann,
/// center=True) on min(100, exp(mag)) * e^{i sin(phase)}. Output is trimmed
/// to exactly T_mel * 480 samples and clamped to ±AUDIO_LIMIT.
fn istft(spec: &[Vec<f32>], t_mel: usize) -> Vec<f32> {
    let t_stft = spec[0].len();
    let n_samples = (t_stft - 1) * ISTFT_HOP + ISTFT_N_FFT;
    let mut wav = vec![0.0f32; n_samples];
    let mut win_sum = vec![0.0f32; n_samples];
    let win = periodic_hann(ISTFT_N_FFT);

    for frame in 0..t_stft {
        let mut re = [0.0f32; N_FREQ];
        let mut im = [0.0f32; N_FREQ];
        for f in 0..N_FREQ {
            let mag = spec[f][frame].exp().min(100.0);
            let ph = spec[N_FREQ + f][frame].sin();
            re[f] = mag * ph.cos();
            im[f] = mag * ph.sin();
        }
        let start = frame * ISTFT_HOP;
        // Half-spectrum IDFT, 2x on interior bins (implicit conjugate symmetry).
        for n in 0..ISTFT_N_FFT {
            if start + n >= n_samples {
                break;
            }
            let mut sample = re[0];
            for f in 1..N_FREQ - 1 {
                let angle = 2.0 * std::f32::consts::PI * (f * n) as f32 / ISTFT_N_FFT as f32;
                sample += 2.0 * (re[f] * angle.cos() - im[f] * angle.sin());
            }
            let angle = 2.0 * std::f32::consts::PI * ((N_FREQ - 1) * n) as f32 / ISTFT_N_FFT as f32;
            sample += re[N_FREQ - 1] * angle.cos() - im[N_FREQ - 1] * angle.sin();
            sample /= ISTFT_N_FFT as f32;
            wav[start + n] += sample * win[n];
            win_sum[start + n] += win[n] * win[n];
        }
    }
    for i in 0..n_samples {
        if win_sum[i] > 1e-8 {
            wav[i] /= win_sum[i];
        }
    }
    // center=True: drop n_fft/2 head samples; zero-pad/truncate to T_mel*480.
    let center_pad = ISTFT_N_FFT / 2;
    (0..t_mel * UPSAMPLE_SCALE)
        .map(|i| {
            let v = wav.get(center_pad + i).copied().unwrap_or(0.0);
            v.clamp(-AUDIO_LIMIT, AUDIO_LIMIT)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Needs the real weights under ./models; not part of the CPU-only suite.
    // Run: cargo test hift_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn hift_smoke() {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        let hift = Hift::load("models/Fun-CosyVoice3-0.5B-2512", &device).unwrap();
        let t_mel = 25usize;
        let mel = Tensor::randn(0f32, 1f32, (t_mel, MEL_DIM), &Device::Cpu).unwrap();
        let wav = hift.mel_to_waveform(&mel).unwrap();
        assert_eq!(wav.len(), t_mel * UPSAMPLE_SCALE);
        assert!(wav.iter().all(|v| v.is_finite()));
        let rms = (wav.iter().map(|v| v * v).sum::<f32>() / wav.len() as f32).sqrt();
        let peak = wav.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        eprintln!(
            "hift smoke: {} samples, rms={rms:.4}, peak={peak:.4}",
            wav.len()
        );
    }
}
