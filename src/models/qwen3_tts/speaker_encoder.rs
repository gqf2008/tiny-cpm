//! Qwen3-TTS speaker encoder (ECAPA-TDNN), ported from
//! github.com/QwenLM/Qwen3-TTS qwen_tts/core/models/modeling_qwen3_tts.py
//! (`Res2NetBlock` / `SqueezeExcitationBlock` / `AttentiveStatisticsPooling` /
//! `TimeDelayNetBlock` / `SqueezeExcitationRes2NetBlock` / `Qwen3TTSSpeakerEncoder`
//! / `mel_spectrogram`). Used only for voice cloning: turns a 24 kHz reference
//! waveform into one raw 2048-d speaker embedding (no L2 norm).
//!
//! Front-end (slaney mel, matches `mel_spectrogram`): reflect-pad 384 → STFT
//! (n_fft 1024, hop 256, periodic Hann, center=False) → magnitude sqrt(power+1e-9)
//! → 128-bin Slaney mel (fmax 12000) → log(clamp 1e-5). Network: conv 128→512 k5
//! → 3× SE-Res2Net (tdnn1 k1 → Res2Net scale8 dil{2,3,4} → tdnn2 k1 → SE 512→128→512
//! → residual) → mfa conv 1536→1536 k1 → ASP (concat[x,mean,std]→128→tanh→1536→
//! softmax → weighted mean⊕std = 3072) → fc 3072→2048. Every conv is
//! `padding="same", padding_mode="reflect"` — implemented as manual reflect-pad
//! ((k-1)*dil/2 each side) + a plain padding-0 conv.
use anyhow::{Result, anyhow};
use candle_core::{D, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder, conv1d};

use crate::models::cosyvoice3::s3tok::periodic_hann;
use crate::models::qwen3_tts::config::SpeakerEncoderParams;
use crate::utils::audio_utils::{MelScale, mel_filter_bank, torch_stft};
use crate::utils::tensor_utils::pad_reflect_last_dim;

/// A conv1d with PyTorch `padding="same", padding_mode="reflect"`: manual reflect
/// pad of `((k-1)*dil)/2` on both sides, then a padding-0 conv (k odd → output
/// length == input length).
#[derive(Debug, Clone)]
struct ReflectConv {
    conv: Conv1d,
    pad: usize,
}

impl ReflectConv {
    fn new(vb: VarBuilder, in_c: usize, out_c: usize, k: usize, dilation: usize) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(in_c, out_c, k, cfg, vb)?;
        Ok(Self {
            conv,
            pad: (k - 1) * dilation / 2,
        })
    }
    /// x: (B, C, T)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.pad > 0 {
            pad_reflect_last_dim(x, (self.pad, self.pad))?
        } else {
            x.clone()
        };
        Ok(self.conv.forward(&x)?)
    }
}

/// `TimeDelayNetBlock`: conv → ReLU.
#[derive(Debug, Clone)]
struct TimeDelayNetBlock {
    conv: ReflectConv,
}

impl TimeDelayNetBlock {
    fn new(vb: VarBuilder, in_c: usize, out_c: usize, k: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            conv: ReflectConv::new(vb.pp("conv"), in_c, out_c, k, dilation)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.conv.forward(x)?.relu()?)
    }
}

/// `Res2NetBlock`: split channels into `scale` chunks; chunk 0 passthrough, chunk i≥1
/// through tdnn block i-1 (with running residual add from chunk 2 on).
struct Res2NetBlock {
    blocks: Vec<TimeDelayNetBlock>, // scale-1 of them
    scale: usize,
}

impl Res2NetBlock {
    fn new(
        vb: VarBuilder,
        channels: usize,
        scale: usize,
        k: usize,
        dilation: usize,
    ) -> Result<Self> {
        let sub = channels / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..scale - 1 {
            blocks.push(TimeDelayNetBlock::new(
                vb.pp("blocks").pp(i),
                sub,
                sub,
                k,
                dilation,
            )?);
        }
        Ok(Self { blocks, scale })
    }
    /// x: (B, C, T)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, c, _t) = x.dims3()?;
        let sub = c / self.scale;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        let mut prev: Option<Tensor> = None;
        for i in 0..self.scale {
            let part = x.narrow(1, i * sub, sub)?;
            let out = if i == 0 {
                part
            } else if i == 1 {
                let o = self.blocks[i - 1].forward(&part)?;
                prev = Some(o.clone());
                o
            } else {
                let p = prev.ok_or_else(|| anyhow!("res2net: missing prev"))?;
                let o = self.blocks[i - 1].forward(&(part + p)?)?;
                prev = Some(o.clone());
                o
            };
            outputs.push(out);
        }
        Ok(Tensor::cat(&outputs, 1)?)
    }
}

/// `SqueezeExcitationBlock`: global mean over time → conv(in→se) → ReLU → conv(se→out)
/// → sigmoid → scale input.
struct SqueezeExcitationBlock {
    conv1: ReflectConv,
    conv2: ReflectConv,
}

impl SqueezeExcitationBlock {
    fn new(vb: VarBuilder, in_c: usize, se_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            conv1: ReflectConv::new(vb.pp("conv1"), in_c, se_c, 1, 1)?,
            conv2: ReflectConv::new(vb.pp("conv2"), se_c, out_c, 1, 1)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(2)?; // (B, C, 1)
        let s = self.conv1.forward(&mean)?.relu()?;
        let s = candle_nn::ops::sigmoid(&self.conv2.forward(&s)?)?;
        Ok(x.broadcast_mul(&s)?)
    }
}

/// `SqueezeExcitationRes2NetBlock`: tdnn1 → res2net → tdnn2 → se → + residual.
struct SeRes2NetBlock {
    tdnn1: TimeDelayNetBlock,
    res2net: Res2NetBlock,
    tdnn2: TimeDelayNetBlock,
    se: SqueezeExcitationBlock,
}

impl SeRes2NetBlock {
    fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        scale: usize,
        se_c: usize,
        k: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: TimeDelayNetBlock::new(vb.pp("tdnn1"), in_c, out_c, 1, 1)?,
            res2net: Res2NetBlock::new(vb.pp("res2net_block"), out_c, scale, k, dilation)?,
            tdnn2: TimeDelayNetBlock::new(vb.pp("tdnn2"), out_c, out_c, 1, 1)?,
            se: SqueezeExcitationBlock::new(vb.pp("se_block"), out_c, se_c, out_c)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.tdnn1.forward(x)?;
        let h = self.res2net.forward(&h)?;
        let h = self.tdnn2.forward(&h)?;
        let h = self.se.forward(&h)?;
        Ok((h + residual)?)
    }
}

/// `AttentiveStatisticsPooling`. The mask is all-ones (single full-length utterance),
/// so the global mean/std reduce to plain temporal statistics; the attention softmax
/// is over time.
struct AttentiveStatisticsPooling {
    tdnn: TimeDelayNetBlock, // 3*channels → attention_channels, k1
    conv: ReflectConv,       // attention_channels → channels, k1
    channels: usize,
}

impl AttentiveStatisticsPooling {
    fn new(vb: VarBuilder, channels: usize, attention_channels: usize) -> Result<Self> {
        Ok(Self {
            tdnn: TimeDelayNetBlock::new(vb.pp("tdnn"), channels * 3, attention_channels, 1, 1)?,
            conv: ReflectConv::new(vb.pp("conv"), attention_channels, channels, 1, 1)?,
            channels,
        })
    }
    /// attention-weighted (over time) mean and std of x; x,m: (B, C, T).
    fn weighted_stats(x: &Tensor, m: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = x.broadcast_mul(m)?.sum(2)?; // (B, C)
        let var = x
            .broadcast_sub(&mean.unsqueeze(2)?)?
            .sqr()?
            .broadcast_mul(m)?
            .sum(2)?; // (B, C)
        let std = var.clamp(1e-12, f64::INFINITY)?.sqrt()?;
        Ok((mean, std))
    }
    /// x: (B, C, T) → (B, 2C, 1)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, t) = x.dims3()?;
        // global (uniform) mean/std, broadcast back over time.
        let gmean = x.mean(2)?; // (B, C)
        let gvar = x.broadcast_sub(&gmean.unsqueeze(2)?)?.sqr()?.mean(2)?;
        let gstd = gvar.clamp(1e-12, f64::INFINITY)?.sqrt()?;
        let gmean = gmean.unsqueeze(2)?.broadcast_as((b, self.channels, t))?;
        let gstd = gstd.unsqueeze(2)?.broadcast_as((b, self.channels, t))?;
        let attn_in = Tensor::cat(&[x, &gmean, &gstd], 1)?; // (B, 3C, T)
        let attn = self.conv.forward(&self.tdnn.forward(&attn_in)?.tanh()?)?; // (B, C, T)
        let attn = candle_nn::ops::softmax(&attn, 2)?; // attention weights over time
        let (mean, std) = Self::weighted_stats(x, &attn)?;
        let pooled = Tensor::cat(&[&mean, &std], 1)?; // (B, 2C)
        Ok(pooled.unsqueeze(2)?)
    }
}

#[allow(clippy::large_enum_variant)] // only 4 blocks; not worth boxing.
enum Block {
    Tdnn(TimeDelayNetBlock),
    SeRes2(SeRes2NetBlock),
}

pub struct SpeakerEncoder {
    params: SpeakerEncoderParams,
    device: Device,
    blocks: Vec<Block>, // [tdnn, se-res2, se-res2, se-res2]
    mfa: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: ReflectConv,
    mel_filters: Tensor, // (n_fft/2+1, mel_dim)
}

impl SpeakerEncoder {
    pub fn new(vb: VarBuilder, params: SpeakerEncoderParams, device: &Device) -> Result<Self> {
        let ch = &params.channels;
        let ks = &params.kernel_sizes;
        let dl = &params.dilations;
        // blocks.0 = initial TDNN (mel_dim → ch[0]); blocks.1..=3 = SE-Res2Net.
        let b0 =
            TimeDelayNetBlock::new(vb.pp("blocks").pp(0), params.mel_dim, ch[0], ks[0], dl[0])?;
        let mut blocks = vec![Block::Tdnn(b0)];
        for i in 1..ch.len() - 1 {
            blocks.push(Block::SeRes2(SeRes2NetBlock::new(
                vb.pp("blocks").pp(i),
                ch[i - 1],
                ch[i],
                params.res2net_scale,
                params.se_channels,
                ks[i],
                dl[i],
            )?));
        }
        let last = ch[ch.len() - 1];
        let mfa =
            TimeDelayNetBlock::new(vb.pp("mfa"), last, last, ks[ks.len() - 1], dl[dl.len() - 1])?;
        let asp = AttentiveStatisticsPooling::new(vb.pp("asp"), last, params.attention_channels)?;
        let fc = ReflectConv::new(vb.pp("fc"), last * 2, params.enc_dim, 1, 1)?;
        let mel_filters = mel_filter_bank(
            1 + params.n_fft / 2,
            params.mel_dim,
            params.fmin,
            params.fmax,
            params.sample_rate as f32,
            Some("slaney"),
            MelScale::Slaney,
            false,
            device,
        )?;
        Ok(Self {
            params,
            device: device.clone(),
            blocks,
            mfa,
            asp,
            fc,
            mel_filters,
        })
    }

    /// wav_24k: mono f32 samples @ 24 kHz → raw speaker embedding (enc_dim,).
    pub fn embed(&self, wav_24k: &[f32]) -> Result<Tensor> {
        let mel = self.mel_spectrogram(wav_24k)?; // (T, mel_dim)
        let mut h = mel.unsqueeze(0)?.transpose(1, 2)?; // (1, mel_dim, T)
        let mut feats: Vec<Tensor> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            h = match block {
                Block::Tdnn(b) => b.forward(&h)?,
                Block::SeRes2(b) => b.forward(&h)?,
            };
            feats.push(h.clone());
            if std::env::var("QWEN3_TTS_DUMP_MEL").is_ok() {
                let s = h.flatten_all()?.to_vec1::<f32>()?;
                let (mn, mx) = (
                    s.iter().copied().fold(f32::INFINITY, f32::min),
                    s.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                );
                let mean = s.iter().sum::<f32>() / s.len() as f32;
                eprintln!(
                    "CANDLE_BLK{} mean={mean:.4} max={mx:.4} min={mn:.4}",
                    feats.len() - 1
                );
            }
        }
        // multi-layer feature aggregation: concat blocks 1..=3, then mfa.
        let agg = Tensor::cat(&feats[1..], 1)?;
        let h = self.mfa.forward(&agg)?;
        if std::env::var("QWEN3_TTS_DUMP_MEL").is_ok() {
            let s = h.flatten_all()?.to_vec1::<f32>()?;
            let mean = s.iter().sum::<f32>() / s.len() as f32;
            eprintln!("CANDLE_BLK4_mfa mean={mean:.4}");
        }
        let h = self.asp.forward(&h)?; // (1, 2C, 1)
        if std::env::var("QWEN3_TTS_DUMP_MEL").is_ok() {
            let s = h.flatten_all()?.to_vec1::<f32>()?;
            let (mn, mx) = (
                s.iter().copied().fold(f32::INFINITY, f32::min),
                s.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            );
            let mean = s.iter().sum::<f32>() / s.len() as f32;
            eprintln!("CANDLE_BLK5_asp mean={mean:.4} max={mx:.4} min={mn:.4}");
        }
        let h = self.fc.forward(&h)?; // (1, enc_dim, 1)
        Ok(h.squeeze(0)?.squeeze(D::Minus1)?) // (enc_dim,)
    }

    /// `mel_spectrogram`: reflect-pad → STFT → magnitude → slaney mel → log-clamp.
    /// Returns (T, mel_dim).
    fn mel_spectrogram(&self, wav_24k: &[f32]) -> Result<Tensor> {
        anyhow::ensure!(!wav_24k.is_empty(), "speaker_encoder: empty waveform");
        let p = &self.params;
        let pad = (p.n_fft - p.hop_length) / 2; // 384
        let mut pcm = wav_24k.to_vec();
        if pcm.len() <= pad {
            pcm.resize(pad + 1, 0.0);
        }
        let n = pcm.len();
        let wav = Tensor::from_vec(pcm, (1, n), &self.device)?;
        let padded = pad_reflect_last_dim(&wav, (pad, pad))?;
        let window = periodic_hann(p.n_fft, &self.device)?.reshape((1, 1, p.n_fft))?;
        let power = torch_stft(&padded, p.n_fft, p.hop_length, &window)?; // (1, T, n_fft/2+1)
        let mag = (power + 1e-9)?.sqrt()?;
        let mel = mag.broadcast_matmul(&self.mel_filters)?; // (1, T, mel_dim)
        let mel = mel.clamp(1e-5f32, f32::INFINITY)?.log()?;
        if std::env::var("QWEN3_TTS_DUMP_MEL").is_ok() {
            let m: Vec<f32> = mel.flatten_all()?.to_vec1::<f32>()?;
            eprintln!("CANDLE_MEL {:?}", m);
        }
        Ok(mel.squeeze(0)?)
    }
}
