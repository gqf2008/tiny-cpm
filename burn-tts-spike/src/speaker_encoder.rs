//! ECAPA-TDNN speaker encoder (voice cloning), ported 1:1 from tiny-cpm
//! src/models/qwen3_tts/speaker_encoder.rs. Turns a 24 kHz reference waveform
//! into one raw enc_dim-d speaker embedding (no L2 norm).
//!
//! Front-end runs on the CPU (audio::speaker_mel: reflect-pad 384 → STFT
//! 1024/256 periodic Hann → sqrt(power+1e-9) → 128-bin slaney mel @ 24 kHz →
//! log(clamp 1e-5)); the network runs in F32 like the candle reference. Every
//! conv is `padding="same", padding_mode="reflect"` — manual reflect-pad
//! ((k-1)*dil/2 each side) + a plain padding-0 conv.

use anyhow::Result;
use burn::tensor::{DType, Float, Tensor, ops::ConvOptions};

use crate::config::SpeakerEncoderParams;
use crate::model::Weights;

/// Speaker encoder dtype: F32, faithful to candle (weights bf16 in the
/// checkpoint are converted to F32 on load, exactly like candle's F32 mmap).
const SPK_DT: DType = DType::F32;

/// A conv1d with PyTorch `padding="same", padding_mode="reflect"`: manual
/// reflect pad of `((k-1)*dil)/2` on both sides, then a padding-0 conv.
struct ReflectConv {
    w: Tensor<3>,
    b: Option<Tensor<1>>,
    pad: usize,
    dilation: usize,
}

impl ReflectConv {
    fn new(
        w: &Weights,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            w: w.get(&format!("{prefix}.weight"), SPK_DT)?,
            b: Some(w.get(&format!("{prefix}.bias"), SPK_DT)?),
            pad: (k - 1) * dilation / 2,
            dilation,
        })
    }
    /// x: (B, C, T)
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = if self.pad > 0 {
            let t = x.dims()[2];
            // candle pad_reflect_last_dim: left = flip(x[1..=pad]), right =
            // flip(x[t-pad..t]).
            let left = x.clone().narrow(2, 1, self.pad).flip([2]);
            let right = x.clone().narrow(2, t - self.pad, self.pad).flip([2]);
            Tensor::cat(vec![left, x, right], 2)
        } else {
            x
        };
        let opts = ConvOptions::new([1], [0], [self.dilation], 1);
        burn::tensor::module::conv1d(x, self.w.clone(), self.b.clone(), opts)
    }
}

/// `TimeDelayNetBlock`: conv → ReLU.
struct TimeDelayNetBlock {
    conv: ReflectConv,
}

impl TimeDelayNetBlock {
    fn new(
        w: &Weights,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        k: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv: ReflectConv::new(w, &format!("{prefix}.conv"), in_c, out_c, k, dilation)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        burn::tensor::activation::relu(self.conv.forward(x))
    }
}

/// `Res2NetBlock`: split channels into `scale` chunks; chunk 0 passthrough,
/// chunk i≥1 through tdnn block i-1 (with running residual add from chunk 2 on).
struct Res2NetBlock {
    blocks: Vec<TimeDelayNetBlock>, // scale-1 of them
    scale: usize,
}

impl Res2NetBlock {
    fn new(
        w: &Weights,
        prefix: &str,
        channels: usize,
        scale: usize,
        k: usize,
        dilation: usize,
    ) -> Result<Self> {
        let sub = channels / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..scale - 1 {
            blocks.push(TimeDelayNetBlock::new(
                w,
                &format!("{prefix}.blocks.{i}"),
                sub,
                sub,
                k,
                dilation,
            )?);
        }
        Ok(Self { blocks, scale })
    }
    /// x: (B, C, T)
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let c = x.dims()[1];
        let sub = c / self.scale;
        let mut outputs: Vec<Tensor<3>> = Vec::with_capacity(self.scale);
        let mut prev: Option<Tensor<3>> = None;
        for i in 0..self.scale {
            let part = x.clone().narrow(1, i * sub, sub);
            let out = if i == 0 {
                part
            } else if i == 1 {
                let o = self.blocks[i - 1].forward(part);
                prev = Some(o.clone());
                o
            } else {
                let p = prev.take().expect("res2net: missing prev");
                let o = self.blocks[i - 1].forward(part + p);
                prev = Some(o.clone());
                o
            };
            outputs.push(out);
        }
        Tensor::cat(outputs, 1)
    }
}

/// `SqueezeExcitationBlock`: global mean over time → conv(in→se) → ReLU →
/// conv(se→out) → sigmoid → scale input.
struct SqueezeExcitationBlock {
    conv1: ReflectConv,
    conv2: ReflectConv,
}

impl SqueezeExcitationBlock {
    fn new(w: &Weights, prefix: &str, in_c: usize, se_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            conv1: ReflectConv::new(w, &format!("{prefix}.conv1"), in_c, se_c, 1, 1)?,
            conv2: ReflectConv::new(w, &format!("{prefix}.conv2"), se_c, out_c, 1, 1)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mean = x.clone().mean_dim(2); // (B, C, 1)
        let s = burn::tensor::activation::relu(self.conv1.forward(mean));
        let s = burn::tensor::activation::sigmoid(self.conv2.forward(s));
        x * s
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
        w: &Weights,
        prefix: &str,
        in_c: usize,
        out_c: usize,
        scale: usize,
        se_c: usize,
        k: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: TimeDelayNetBlock::new(w, &format!("{prefix}.tdnn1"), in_c, out_c, 1, 1)?,
            res2net: Res2NetBlock::new(
                w,
                &format!("{prefix}.res2net_block"),
                out_c,
                scale,
                k,
                dilation,
            )?,
            tdnn2: TimeDelayNetBlock::new(w, &format!("{prefix}.tdnn2"), out_c, out_c, 1, 1)?,
            se: SqueezeExcitationBlock::new(w, &format!("{prefix}.se_block"), out_c, se_c, out_c)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let residual = x.clone();
        let h = self.tdnn1.forward(x);
        let h = self.res2net.forward(h);
        let h = self.tdnn2.forward(h);
        let h = self.se.forward(h);
        h + residual
    }
}

/// `AttentiveStatisticsPooling`. The mask is all-ones (single full-length
/// utterance), so the global mean/std reduce to plain temporal statistics.
struct AttentiveStatisticsPooling {
    tdnn: TimeDelayNetBlock, // 3*channels → attention_channels, k1
    conv: ReflectConv,       // attention_channels → channels, k1
    channels: usize,
}

impl AttentiveStatisticsPooling {
    fn new(w: &Weights, channels: usize, attention_channels: usize) -> Result<Self> {
        Ok(Self {
            tdnn: TimeDelayNetBlock::new(
                w,
                "speaker_encoder.asp.tdnn",
                channels * 3,
                attention_channels,
                1,
                1,
            )?,
            conv: ReflectConv::new(
                w,
                "speaker_encoder.asp.conv",
                attention_channels,
                channels,
                1,
                1,
            )?,
            channels,
        })
    }
    /// attention-weighted (over time) mean and std of x; x: (B, C, T).
    /// var = Σ (x-mean)²·m — the weight is applied AFTER squaring (candle
    /// broadcast_mul(m) on the squared residual), NOT to the residual itself.
    fn weighted_stats(x: Tensor<3>, m: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let mean = x.clone() * m.clone();
        let mean = mean.sum_dim(2); // (B, C, 1)
        let var = (x - mean.clone()).powf_scalar(2.0) * m;
        let var = var.sum_dim(2); // (B, C, 1)
        let std = var.clamp_min(1e-12).sqrt();
        (mean, std)
    }
    /// x: (B, C, T) → (B, 2C, 1)
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [_, c, t] = x.dims();
        // global (uniform) mean/std, broadcast back over time.
        let gmean = x.clone().mean_dim(2); // (B, C, 1)
        let gvar = (x.clone() - gmean.clone()).powf_scalar(2.0).mean_dim(2);
        let gstd = gvar.clamp_min(1e-12).sqrt();
        let gmean = gmean.clone().repeat_dim(2, t); // broadcast_as (B,C,T)
        let gstd = gstd.repeat_dim(2, t);
        let attn_in = Tensor::cat(vec![x.clone(), gmean, gstd], 1); // (B, 3C, T)
        let attn = self
            .conv
            .forward(burn::tensor::activation::tanh(self.tdnn.forward(attn_in)));
        let attn = burn::tensor::activation::softmax(attn, 2); // attention over time
        let (mean, std) = Self::weighted_stats(x, attn);
        let pooled = Tensor::cat(vec![mean, std], 1); // (B, 2C, 1)
        pooled
    }
}

enum Block {
    Tdnn(TimeDelayNetBlock),
    SeRes2(SeRes2NetBlock),
}

pub struct SpeakerEncoder {
    params: SpeakerEncoderParams,
    blocks: Vec<Block>, // [tdnn, se-res2, se-res2, se-res2]
    mfa: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: ReflectConv,
}

impl SpeakerEncoder {
    pub fn new(w: &Weights, params: SpeakerEncoderParams) -> Result<Self> {
        let ch = &params.channels;
        let ks = &params.kernel_sizes;
        let dl = &params.dilations;
        // blocks.0 = initial TDNN (mel_dim → ch[0]); blocks.1..=3 = SE-Res2Net.
        let b0 = TimeDelayNetBlock::new(
            w,
            "speaker_encoder.blocks.0",
            params.mel_dim,
            ch[0],
            ks[0],
            dl[0],
        )?;
        let mut blocks = vec![Block::Tdnn(b0)];
        for i in 1..ch.len() - 1 {
            blocks.push(Block::SeRes2(SeRes2NetBlock::new(
                w,
                &format!("speaker_encoder.blocks.{i}"),
                ch[i - 1],
                ch[i],
                params.res2net_scale,
                params.se_channels,
                ks[i],
                dl[i],
            )?));
        }
        let last = ch[ch.len() - 1];
        let mfa = TimeDelayNetBlock::new(
            w,
            "speaker_encoder.mfa",
            last,
            last,
            ks[ks.len() - 1],
            dl[dl.len() - 1],
        )?;
        let asp = AttentiveStatisticsPooling::new(w, last, params.attention_channels)?;
        let fc = ReflectConv::new(w, "speaker_encoder.fc", last * 2, params.enc_dim, 1, 1)?;
        Ok(Self {
            params,
            blocks,
            mfa,
            asp,
            fc,
        })
    }

    /// wav_24k: mono f32 samples @ 24 kHz → raw speaker embedding (enc_dim,).
    pub fn embed(&self, wav_24k: &[f32], device: &burn::tensor::Device) -> Result<Tensor<1>> {
        let mel = crate::audio::speaker_mel(
            wav_24k,
            self.params.n_fft,
            self.params.hop_length,
            self.params.mel_dim,
        ); // (T, mel_dim)
        let t = mel.len() / self.params.mel_dim;
        let h: Tensor<3> = Tensor::<1, Float>::from_floats(mel.as_slice(), device)
            .reshape([t, self.params.mel_dim])
            .swap_dims(0, 1)
            .unsqueeze(); // (1, mel_dim, T)
        let mut h = h;
        let mut feats: Vec<Tensor<3>> = Vec::with_capacity(self.blocks.len());
        for (bi, block) in self.blocks.iter().enumerate() {
            h = match block {
                Block::Tdnn(b) => b.forward(h),
                Block::SeRes2(b) => b.forward(h),
            };
            feats.push(h.clone());
        }
        // multi-layer feature aggregation: concat blocks 1..=3, then mfa.
        let agg = Tensor::cat(feats[1..].to_vec(), 1);
        let h = self.mfa.forward(agg);
        let h = self.asp.forward(h); // (1, 2C, 1)
        let h = self.fc.forward(h); // (1, enc_dim, 1)
        Ok(h.squeeze_dim::<2>(0).squeeze_dim::<1>(1)) // (enc_dim,)
    }
}
