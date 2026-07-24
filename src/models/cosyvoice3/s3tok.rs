//! CosyVoice3 s3tok (see mod.rs header).
//!
//! Speech tokenizer V3: 16 kHz mono ref wav -> 128-mel whisper-style frontend
//! -> 2x Conv1d subsampler (x4) -> 12 FSMN-augmented attention blocks ->
//! Linear(1280->8) -> 8-axis FSQ (3 levels/axis) -> token ids in [0, 6561)
//! at 25 Hz.
//!
//! Ported from CrispASR's `cv3_compute_s3tok_log_mel` /
//! `cv3_build_s3tok_graph` / `cv3_tokenize_s3tok` (src/cosyvoice3_tts.cpp).
//! Weights come from the GGUF F16 conversion
//! (models/convert-cosyvoice3-s3tok-to-gguf.py), keys prefixed
//! `cosyvoice3.s3tok.`; candle reverses the ggml dims, so dequantized
//! tensors are already in PyTorch layout (conv1d (OC, IC, KW), linear (out, in)).

use std::path::Path;

use anyhow::{Result, anyhow};
use candle_core::quantized::gguf_file;
use candle_core::{D, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module};

use crate::common::modules::{eager_attention_forward, log10};
use crate::position_embed::rope::{RoPE, apply_rotary_pos_emb};
use crate::utils::audio_utils::{MelScale, mel_filter_bank, torch_stft};
use crate::utils::tensor_utils::pad_reflect_last_dim;

// Frontend (16 kHz, 100 fps, whisper-style log10 mel).
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;

// Encoder defaults (GGUF metadata overrides these when present).
const N_LAYERS: usize = 12;
const D_MODEL: usize = 1280;
const N_HEADS: usize = 20;
const HEAD_DIM: usize = D_MODEL / N_HEADS; // 64
const ROPE_THETA: f32 = 10000.0;

// FSQ: 8 axes, 3 levels each.
const FSQ_DIM: usize = 8;
const FSQ_GAIN: f32 = 0.9990000128746033;

/// Periodic Hann window (`torch.hann_window(N)` default `periodic=True`).
pub(crate) fn periodic_hann(n: usize, device: &Device) -> Result<Tensor> {
    let w: Vec<f32> = (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect();
    Ok(Tensor::from_vec(w, n, device)?)
}

/// Locate the single GGUF in `dir` whose filename contains `marker`.
pub(crate) fn find_gguf(dir: &Path, marker: &str) -> Result<std::path::PathBuf> {
    let mut hits = vec![];
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.contains(marker) && name.ends_with(".gguf") {
            hits.push(path);
        }
    }
    match hits.len() {
        1 => Ok(hits.pop().unwrap()),
        0 => Err(anyhow!(
            "no *{marker}*.gguf found in {}; run the GGUF conversion first",
            dir.display()
        )),
        _ => Err(anyhow!(
            "multiple *{marker}*.gguf files in {}: {hits:?}",
            dir.display()
        )),
    }
}

/// Load + dequantize one GGUF tensor (F16 -> F32) onto `device`.
pub(crate) fn gguf_tensor<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    name: &str,
) -> Result<Tensor> {
    Ok(ct.tensor(reader, name, device)?.dequantize(device)?)
}

/// Thin wrapper bundling a GGUF `Content`, its reader and the target device,
/// so model loaders don't fight the borrow checker over closures.
pub(crate) struct GgufReader<'a, R: std::io::Seek + std::io::Read> {
    ct: &'a gguf_file::Content,
    reader: &'a mut R,
    device: &'a Device,
}

impl<'a, R: std::io::Seek + std::io::Read> GgufReader<'a, R> {
    pub(crate) fn new(ct: &'a gguf_file::Content, reader: &'a mut R, device: &'a Device) -> Self {
        Self { ct, reader, device }
    }

    pub(crate) fn tensor(&mut self, name: &str) -> Result<Tensor> {
        gguf_tensor(self.ct, self.reader, self.device, name)
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    /// Tensor if present, None otherwise (optional biases).
    pub(crate) fn opt_tensor(&mut self, name: &str) -> Result<Option<Tensor>> {
        if self.has(name) {
            Ok(Some(self.tensor(name)?))
        } else {
            Ok(None)
        }
    }
}

fn gguf_u32(ct: &gguf_file::Content, key: &str, default: usize) -> usize {
    ct.metadata
        .get(key)
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(default)
}

struct S3TokBlock {
    attn_ln: LayerNorm, // eps 1e-6
    q_proj: Linear,     // biased
    k_proj: Linear,     // UNBIASED
    v_proj: Linear,     // biased
    o_proj: Linear,     // biased
    fsmn_w: Tensor,     // depthwise (C, 1, K), no bias
    mlp_ln: LayerNorm,  // eps 1e-5
    mlp_up: Linear,
    mlp_dn: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl S3TokBlock {
    /// Depthwise Conv1d(k=31, pad=15) on (1, C, T), implemented as an unfold
    /// (31 shifted narrows + weighted sum) instead of grouped conv: Metal
    /// grouped conv1d decomposes into C per-group calls, which is 1280
    /// kernel launches per block.
    fn fsmn(&self, v_t: &Tensor) -> Result<Tensor> {
        let k = self.fsmn_w.dim(2)?;
        let pad = k / 2;
        let t = v_t.dim(2)?;
        let x_pad = v_t.pad_with_zeros(2, pad, pad)?; // (1, C, T + 2*pad)
        let cols: Vec<Tensor> = (0..k)
            .map(|i| x_pad.narrow(2, i, t))
            .collect::<std::result::Result<_, _>>()?;
        let cols = Tensor::stack(&cols, 2)?; // (1, C, K, T)
        // weight (C, 1, K) -> (1, C, K, 1)
        let w = self.fsmn_w.unsqueeze(0)?.transpose(2, 3)?;
        Ok(cols.broadcast_mul(&w)?.sum(2)?) // (1, C, T)
    }

    /// x: (1, T, D); cos/sin: (T, head_dim) NEOX tables.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let residual = x.clone();
        let h = self.attn_ln.forward(x)?;
        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;

        // FSMN memory block: depthwise conv on V + residual of V, added to
        // the attention output (before the outer residual).
        let v_t = v.transpose(1, 2)?; // (1, D, T)
        let fsmn = (self.fsmn(&v_t)? + &v_t)?.transpose(1, 2)?; // (1, T, D)

        let q = q
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin, false)?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn = eager_attention_forward(&q, &k, &v, None, None, scale)?; // (b, T, n_heads, hd)
        let attn = attn.reshape((b, t, d))?;
        let attn = (self.o_proj.forward(&attn)? + &fsmn)?;
        let x = (residual + attn)?;

        let residual = x.clone();
        let h = self.mlp_ln.forward(&x)?;
        let h = self.mlp_dn.forward(&self.mlp_up.forward(&h)?.gelu_erf()?)?;
        Ok((residual + h)?)
    }
}

pub struct S3Tok {
    device: Device,
    mel_filters: Tensor, // (n_mels, n_fft/2+1)
    window: Tensor,      // (1, 1, n_fft) periodic Hann
    conv0: Conv1d,       // 128 -> D, k=3, s=2, p=1
    conv1: Conv1d,       // D -> D, k=3, s=2, p=1
    blocks: Vec<S3TokBlock>,
    fsq_proj: Linear, // D -> 8
    rope: RoPE,
}

impl S3Tok {
    /// Load from the Fun-CosyVoice3 model dir (expects exactly one
    /// `*s3tok*.gguf` inside, e.g. `cosyvoice3-s3tok-f16.gguf`).
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let gguf_path = find_gguf(dir.as_ref(), "s3tok")?;
        let mut reader = std::io::BufReader::new(std::fs::File::open(&gguf_path)?);
        let ct = gguf_file::Content::read(&mut reader)?;

        let n_layers = gguf_u32(&ct, "cosyvoice3.s3tok.n_blocks", N_LAYERS);
        let d_model = gguf_u32(&ct, "cosyvoice3.s3tok.d_model", D_MODEL);
        let n_heads = gguf_u32(&ct, "cosyvoice3.s3tok.n_heads", N_HEADS);
        let head_dim = d_model / n_heads;

        let mut g = GgufReader::new(&ct, &mut reader, device);
        let conv_cfg = Conv1dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv0 = Conv1d::new(
            g.tensor("cosyvoice3.s3tok.subsample.conv0.w")?,
            g.opt_tensor("cosyvoice3.s3tok.subsample.conv0.b")?,
            conv_cfg,
        );
        let conv1 = Conv1d::new(
            g.tensor("cosyvoice3.s3tok.subsample.conv1.w")?,
            g.opt_tensor("cosyvoice3.s3tok.subsample.conv1.b")?,
            conv_cfg,
        );

        let mut blocks = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let p = format!("cosyvoice3.s3tok.blk.{il}");
            blocks.push(S3TokBlock {
                attn_ln: LayerNorm::new(
                    g.tensor(&format!("{p}.attn_ln.w"))?,
                    g.tensor(&format!("{p}.attn_ln.b"))?,
                    1e-6,
                ),
                q_proj: Linear::new(
                    g.tensor(&format!("{p}.attn_q.w"))?,
                    g.opt_tensor(&format!("{p}.attn_q.b"))?,
                ),
                k_proj: Linear::new(g.tensor(&format!("{p}.attn_k.w"))?, None),
                v_proj: Linear::new(
                    g.tensor(&format!("{p}.attn_v.w"))?,
                    g.opt_tensor(&format!("{p}.attn_v.b"))?,
                ),
                o_proj: Linear::new(
                    g.tensor(&format!("{p}.attn_o.w"))?,
                    g.opt_tensor(&format!("{p}.attn_o.b"))?,
                ),
                fsmn_w: g.tensor(&format!("{p}.attn.fsmn_block.w"))?,
                mlp_ln: LayerNorm::new(
                    g.tensor(&format!("{p}.mlp_ln.w"))?,
                    g.tensor(&format!("{p}.mlp_ln.b"))?,
                    1e-5,
                ),
                mlp_up: Linear::new(
                    g.tensor(&format!("{p}.mlp_up.w"))?,
                    g.opt_tensor(&format!("{p}.mlp_up.b"))?,
                ),
                mlp_dn: Linear::new(
                    g.tensor(&format!("{p}.mlp_dn.w"))?,
                    g.opt_tensor(&format!("{p}.mlp_dn.b"))?,
                ),
                n_heads,
                head_dim,
            });
        }
        let fsq_proj = Linear::new(
            g.tensor("cosyvoice3.s3tok.fsq.proj.w")?,
            g.opt_tensor("cosyvoice3.s3tok.fsq.proj.b")?,
        );

        // Slaney filterbank, same math as the whisper frontend but 128 bins
        // and log10 + global clip-max normalization (see encode()).
        let mel_filters = mel_filter_bank(
            1 + N_FFT / 2,
            N_MELS,
            0.0,
            8000.0,
            SAMPLE_RATE as f32,
            Some("slaney"),
            MelScale::Slaney,
            false,
            device,
        )?
        .t()?
        .contiguous()?; // Metal matmul rejects strided operands
        let window = periodic_hann(N_FFT, device)?.reshape((1, 1, N_FFT))?;
        let rope = RoPE::new(head_dim, ROPE_THETA, device)?;

        Ok(Self {
            device: device.clone(),
            mel_filters,
            window,
            conv0,
            conv1,
            blocks,
            fsq_proj,
            rope,
        })
    }

    /// 16 kHz mono PCM -> 128-mel log10 spectrogram (1, n_mels, T), 100 fps.
    /// Power spec, clamp 1e-10, log10, clip at global max - 8, then
    /// (x + 4) / 4, and drop the last frame (cv3_compute_s3tok_log_mel).
    /// Deliberate divergence: we reflect-pad n_fft/2 (upstream torch.stft
    /// center=True default, pad_mode='reflect') where CrispASR zero-pads
    /// (center_pad=true, center_pad_reflect=false) — ours is closer to the
    /// official S3Tokenizer.
    fn log_mel(&self, wav: &Tensor) -> Result<Tensor> {
        let pad = N_FFT / 2;
        let wav = pad_reflect_last_dim(wav, (pad, pad))?;
        let (_, samples) = wav.dims2()?;
        let power = torch_stft(&wav, N_FFT, HOP, &self.window)?.transpose(D::Minus1, D::Minus2)?;
        let n_frames = (samples - N_FFT) / HOP + 1;
        let power = power.narrow(D::Minus1, 0, n_frames - 1)?.contiguous()?;
        let mel = self.mel_filters.broadcast_matmul(&power)?;
        let mel = mel.clamp(1e-10f32, f32::INFINITY)?;
        let log_mel = log10(&mel)?;
        let max_val = log_mel.max_all()?.affine(1.0, -8.0)?;
        let log_mel = log_mel.broadcast_maximum(&max_val)?;
        Ok(log_mel.affine(1.0, 4.0)?.affine(1.0 / 4.0, 0.0)?)
    }

    /// Encoder forward up to the FSQ projection: (1, T, 8).
    fn project(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(mel)?.gelu_erf()?;
        let x = self.conv1.forward(&x)?.gelu_erf()?;
        let x = x.transpose(1, 2)?.contiguous()?; // (1, T/4, D)
        let t_tok = x.dim(1)?;
        let (cos, sin) = self.rope.forward(0, t_tok, &self.device)?;
        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h, &cos, &sin)?;
        }
        Ok(self.fsq_proj.forward(&h)?)
    }

    /// Quantize the FSQ projection to token ids: per axis
    /// `v = clamp(round(tanh(z) * 0.999) + 1, 0, 2)`, `token = sum(v_i * 3^i)`.
    fn fsq_quantize(proj: &Tensor) -> Result<Vec<u32>> {
        let proj = proj.squeeze(0)?.to_vec2::<f32>()?; // (T, 8), on host
        const POWERS: [u32; FSQ_DIM] = [1, 3, 9, 27, 81, 243, 729, 2187];
        let mut out = Vec::with_capacity(proj.len());
        for row in &proj {
            let mut code = 0u32;
            for (i, &z) in row.iter().enumerate() {
                let h = z.tanh() * FSQ_GAIN;
                let v = (h.round_ties_even() as i32 + 1).clamp(0, 2) as u32;
                code += v * POWERS[i];
            }
            out.push(code);
        }
        Ok(out)
    }

    /// 16 kHz mono ref wav -> speech tokens (25 Hz, ids in [0, 6561)).
    pub fn encode(&self, wav_16k: &[f32]) -> Result<Vec<u32>> {
        if wav_16k.is_empty() {
            return Ok(vec![]);
        }
        // reflect pad needs > n_fft/2 samples; zero-pad ultra-short clips.
        let mut pcm;
        let wav: &[f32] = if wav_16k.len() <= N_FFT / 2 {
            pcm = wav_16k.to_vec();
            pcm.resize(N_FFT, 0.0);
            &pcm
        } else {
            wav_16k
        };
        let wav = Tensor::from_vec(wav.to_vec(), (1, wav.len()), &self.device)?;
        let mel = self.log_mel(&wav)?;
        let proj = self.project(&mel)?;
        Self::fsq_quantize(&proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsq_bounds() -> Result<()> {
        // Extreme projections must stay within the 3^8 = 6561 codebook.
        let device = Device::Cpu;
        let proj = Tensor::new(
            vec![
                vec![f32::INFINITY; FSQ_DIM],
                vec![f32::NEG_INFINITY; FSQ_DIM],
            ],
            &device,
        )?
        .unsqueeze(0)?;
        let codes = S3Tok::fsq_quantize(&proj)?;
        assert_eq!(codes[0], 6560);
        assert_eq!(codes[1], 0);
        Ok(())
    }

    #[test]
    fn frontend_shapes() -> Result<()> {
        // 1 s @16k -> 100 mel frames (drop last) -> 25 tokens after x4 subsample.
        let device = Device::Cpu;
        let mel_filters = mel_filter_bank(
            1 + N_FFT / 2,
            N_MELS,
            0.0,
            8000.0,
            SAMPLE_RATE as f32,
            Some("slaney"),
            MelScale::Slaney,
            false,
            &device,
        )?
        .t()?;
        let window = periodic_hann(N_FFT, &device)?.reshape((1, 1, N_FFT))?;
        let s3 = S3Tok {
            device: device.clone(),
            mel_filters,
            window,
            // Unused by log_mel; dummy zero-weight modules.
            conv0: Conv1d::new(
                Tensor::zeros((D_MODEL, N_MELS, 3), candle_core::DType::F32, &device)?,
                None,
                Conv1dConfig {
                    padding: 1,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            conv1: Conv1d::new(
                Tensor::zeros((D_MODEL, D_MODEL, 3), candle_core::DType::F32, &device)?,
                None,
                Conv1dConfig {
                    padding: 1,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            blocks: vec![],
            fsq_proj: Linear::new(
                Tensor::zeros((FSQ_DIM, D_MODEL), candle_core::DType::F32, &device)?,
                None,
            ),
            rope: RoPE::new(HEAD_DIM, ROPE_THETA, &device)?,
        };
        let wav = Tensor::randn(0f32, 0.1, (1, SAMPLE_RATE), &device)?;
        let mel = s3.log_mel(&wav)?;
        assert_eq!(mel.dims(), &[1, N_MELS, 100]);
        let proj = s3.project(&mel)?;
        assert_eq!(proj.dims(), &[1, 25, FSQ_DIM]);
        Ok(())
    }
}
