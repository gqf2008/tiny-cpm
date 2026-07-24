//! CosyVoice3 campplus (see mod.rs header).
//!
//! CAMPPlus speaker encoder: 16 kHz mono ref wav -> Kaldi 80-bin fbank
//! (Povey window, per-utterance mean subtraction) -> FCM 2-D conv head ->
//! TDNN -> 3 CAMDenseTDNN blocks -> StatsPool -> 192-d speaker embedding
//! (used RAW; the flow's spk_affine path applies the L2 norm).
//!
//! Ported from CrispASR's chatterbox_campplus.{h,cpp} (fbank + xvector
//! forward + `compute_prompt_feat_24k`). Weights come from the GGUF F16
//! conversion (models/convert-cosyvoice3-campplus-to-gguf.py), keys prefixed
//! `s3.se.`; candle reverses the ggml dims, so dequantized tensors are
//! already in PyTorch layout (conv2d (OC, IC, KH, KW), conv1d (OC, IC, KW)).

use std::path::Path;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{D, Device, Tensor};
use candle_nn::{BatchNorm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Module, ModuleT};

use super::s3tok::{GgufReader, find_gguf, periodic_hann};
use crate::utils::audio_utils::{
    MelScale, apply_stft, extract_frames, kaldi_get_mel_banks, mel_filter_bank, torch_stft,
};
use crate::utils::tensor_utils::{pad_reflect_last_dim, pad_replicate_last_dim, repeat_interleave};

const BN_EPS: f64 = 1e-5; // PyTorch nn.BatchNorm default

// Kaldi fbank defaults (16 kHz).
const WIN: usize = 400; // 25 ms
const SHIFT: usize = 160; // 10 ms
const PADDED: usize = 512;
const N_MELS: usize = 80;

// CAMDenseTDNN block shape.
const BLOCK_LAYERS: [usize; 3] = [12, 24, 16];
const BLOCK_DILATIONS: [usize; 3] = [1, 2, 2];

// CAM segment pooling (avg_pool1d k=100 s=100, ceil_mode, count_include_pad).
const SEG_LEN: usize = 100;

// 24 kHz prompt mel (Matcha-TTS / CosyVoice mel_spectrogram).
const PROMPT_SR: usize = 24000;
const PROMPT_N_FFT: usize = 1920;
const PROMPT_HOP: usize = 480;
const PROMPT_PAD: usize = (PROMPT_N_FFT - PROMPT_HOP) / 2; // 720
const PROMPT_MAX_SAMPLES: usize = 10 * PROMPT_SR; // DEC_COND_LEN

fn conv2d_cfg(stride: usize) -> Conv2dConfig {
    Conv2dConfig {
        padding: 1,
        stride,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    }
}

/// Eval-mode BN forward (running stats, no update).
fn bn_eval(bn: &BatchNorm, x: &Tensor) -> Result<Tensor> {
    Ok(bn.forward_t(x, false)?)
}

/// Optional BN: None = identity passthrough (CrispASR `fold_bn` null path).
fn bn_eval_opt(bn: &Option<BatchNorm>, x: &Tensor) -> Result<Tensor> {
    match bn {
        Some(bn) => bn_eval(bn, x),
        None => Ok(x.clone()),
    }
}

/// Conv2d with stride (2, 1): candle convs have a square stride, so compute
/// the stride-1 conv and keep every second row along H (valid because the
/// conv is shift-invariant along H).
fn conv2d_stride_h2(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    let y = conv.forward(x)?;
    let h = y.dim(2)?;
    let idx = Tensor::arange_step(0u32, h as u32, 2, y.device())?;
    Ok(y.index_select(&idx, 2)?)
}

/// BatchNorm with affine from `<pfx>.{weight,bias,running_mean,running_var}`.
fn load_bn<R: std::io::Seek + std::io::Read>(
    g: &mut GgufReader<R>,
    pfx: &str,
) -> Result<BatchNorm> {
    let w = g.tensor(&format!("{pfx}.weight"))?;
    let b = g.tensor(&format!("{pfx}.bias"))?;
    let m = g.tensor(&format!("{pfx}.running_mean"))?;
    let v = g.tensor(&format!("{pfx}.running_var"))?;
    Ok(BatchNorm::new(w.dim(0)?, m, v, w, b, BN_EPS)?)
}

/// BatchNorm with affine=False (`running_mean`/`running_var` only).
fn load_bn_no_affine<R: std::io::Seek + std::io::Read>(
    g: &mut GgufReader<R>,
    pfx: &str,
) -> Result<BatchNorm> {
    let m = g.tensor(&format!("{pfx}.running_mean"))?;
    let v = g.tensor(&format!("{pfx}.running_var"))?;
    Ok(BatchNorm::new_no_bias(m.dim(0)?, m, v, BN_EPS)?)
}

/// Head-section BatchNorm, optional. The cstr GGUF revision ships no
/// `s3.se.head.*bn*` tensors at all; CrispASR's loader reads them via
/// try_get and its `fold_bn` falls back to IDENTITY (gamma=1, beta=0) when
/// running_mean/var are absent. Mirror that with None = passthrough.
fn load_bn_opt<R: std::io::Seek + std::io::Read>(
    g: &mut GgufReader<R>,
    pfx: &str,
) -> Result<Option<BatchNorm>> {
    if !g.has(&format!("{pfx}.running_mean")) {
        return Ok(None);
    }
    // running stats present but no weight/bias -> affine=False BN (same
    // fallback as `dense.nl.bn`).
    if !g.has(&format!("{pfx}.weight")) {
        return Ok(Some(load_bn_no_affine(g, pfx)?));
    }
    Ok(Some(load_bn(g, pfx)?))
}

fn load_resblock<R: std::io::Seek + std::io::Read>(
    g: &mut GgufReader<R>,
    p: &str,
    stride_h: usize,
) -> Result<ResBlock> {
    let conv1 = Conv2d::new(
        g.tensor(&format!("{p}.conv1.weight"))?,
        g.opt_tensor(&format!("{p}.conv1.bias"))?,
        conv2d_cfg(1), // stride (2,1) emulated by row subsampling
    );
    let bn1 = load_bn_opt(g, &format!("{p}.bn1"))?;
    let conv2 = Conv2d::new(
        g.tensor(&format!("{p}.conv2.weight"))?,
        g.opt_tensor(&format!("{p}.conv2.bias"))?,
        conv2d_cfg(1),
    );
    let bn2 = load_bn_opt(g, &format!("{p}.bn2"))?;
    let shortcut = if stride_h != 1 {
        let conv = Conv2d::new(
            g.tensor(&format!("{p}.shortcut.0.weight"))?,
            g.opt_tensor(&format!("{p}.shortcut.0.bias"))?,
            Conv2dConfig {
                padding: 0,
                ..conv2d_cfg(1)
            },
        );
        Some((conv, load_bn_opt(g, &format!("{p}.shortcut.1"))?))
    } else {
        None
    };
    Ok(ResBlock {
        conv1,
        bn1,
        conv2,
        bn2,
        shortcut,
        stride_h,
    })
}

fn load_unit<R: std::io::Seek + std::io::Read>(
    g: &mut GgufReader<R>,
    p: &str,
    cfg: Conv1dConfig,
) -> Result<Unit> {
    let conv = Conv1d::new(
        g.tensor(&format!("{p}.linear.weight"))?,
        g.opt_tensor(&format!("{p}.linear.bias"))?,
        cfg,
    );
    Ok(Unit {
        conv,
        bn: load_bn_opt(g, &format!("{p}.nl.bn"))?,
    })
}

/// FCM head BasicResBlock: conv1(s=(stride,1))+BN+ReLU, conv2+BN, plus a
/// 1x1 conv+BN shortcut when downsampling, then ReLU(out + shortcut).
struct ResBlock {
    conv1: Conv2d,
    bn1: Option<BatchNorm>, // None = identity (GGUF revision without head BNs)
    conv2: Conv2d,
    bn2: Option<BatchNorm>,
    shortcut: Option<(Conv2d, Option<BatchNorm>)>,
    stride_h: usize,
}

impl ResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.conv1.forward(x)?;
        if self.stride_h == 2 {
            out = stride2_rows(&out)?;
        }
        let out = bn_eval_opt(&self.bn1, &out)?.relu()?;
        let y = bn_eval_opt(&self.bn2, &self.conv2.forward(&out)?)?;
        let sc = match &self.shortcut {
            Some((conv, bn)) => {
                let mut s = conv.forward(x)?;
                if self.stride_h == 2 {
                    s = stride2_rows(&s)?;
                }
                bn_eval_opt(bn, &s)?
            }
            None => x.clone(),
        };
        Ok((y + sc)?.relu()?)
    }
}

fn stride2_rows(x: &Tensor) -> Result<Tensor> {
    let h = x.dim(2)?;
    let idx = Tensor::arange_step(0u32, h as u32, 2, x.device())?;
    Ok(x.index_select(&idx, 2)?)
}

/// CAMDenseTDNNLayer: BN+ReLU -> 1x1 bottleneck conv -> BN+ReLU -> CAM layer
/// (dilated local conv gated by a sigmoid channel mask from global-mean +
/// 100-frame segment-pool context); the gated output is concatenated onto
/// the layer input (dense block growth).
struct DenseLayer {
    nonl1_bn: BatchNorm,
    l1: Conv1d, // k=1, in -> 128, bias
    // None = identity: the cstr GGUF revision ships no nonl2.bn tensors
    // (CrispASR `fold_bn` null path).
    nonl2_bn: Option<BatchNorm>,
    cam_ll: Conv1d, // k=3, dilation d, pad d, 128 -> 32, no bias
    cam_l1: Conv1d, // k=1, 128 -> 64, bias
    cam_l2: Conv1d, // k=1, 64 -> 32, bias
}

impl DenseLayer {
    /// x: (1, C_in, T) -> (1, C_in + 32, T)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = bn_eval(&self.nonl1_bn, x)?.relu()?;
        let bo = self.l1.forward(&a)?;
        let bo = bn_eval_opt(&self.nonl2_bn, &bo)?.relu()?;

        // CAM local branch.
        let y = self.cam_ll.forward(&bo)?;

        // CAM context: global mean + segment-pooled view, broadcast over T.
        let t = bo.dim(2)?;
        let gmean = bo.mean_keepdim(2)?; // (1, C, 1)
        let n_seg = t.div_ceil(SEG_LEN);
        let bo_pad = bo.pad_with_zeros(2, 0, n_seg * SEG_LEN - t)?.reshape((
            1,
            bo.dim(1)?,
            n_seg,
            SEG_LEN,
        ))?;
        // count_include_pad=True: divisor is the full kernel even in the tail.
        let seg = bo_pad.mean_keepdim(3)?.squeeze(3)?; // (1, C, n_seg)
        let seg_full = repeat_interleave(&seg, SEG_LEN, 2)?.narrow(2, 0, t)?;
        let ctx = gmean.broadcast_add(&seg_full)?;

        let ctx = self.cam_l1.forward(&ctx)?.relu()?;
        let mask = candle_nn::ops::sigmoid(&self.cam_l2.forward(&ctx)?)?;
        let y = y.mul(&mask)?;
        Ok(Tensor::cat(&[x, &y], 1)?)
    }
}

/// Conv1d + BN (+ ReLU applied by the caller) used for tdnn / transit* / dense.
struct Unit {
    conv: Conv1d,
    // None = identity (CrispASR `fold_bn` null path; the cstr GGUF has no
    // tdnn.nl.bn, though transit*/dense BNs are present).
    bn: Option<BatchNorm>,
}

impl Unit {
    /// tdnn order: conv -> BN.
    fn conv_bn(&self, x: &Tensor) -> Result<Tensor> {
        bn_eval_opt(&self.bn, &self.conv.forward(x)?)
    }
    /// transit order: BN -> ReLU -> conv (per the reference `bn_relu_conv1d`).
    fn bn_relu_conv(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.conv.forward(&bn_eval_opt(&self.bn, x)?.relu()?)?)
    }
}

pub struct CampPlus {
    device: Device,
    mel_banks_t: Tensor, // (256, 80) kaldi HTK-style banks
    povey: Tensor,       // (1, 1, 400)
    // FCM head.
    head_conv1: Conv2d,
    head_bn1: Option<BatchNorm>, // None = identity (GGUF without head BNs)
    head_layer1: Vec<ResBlock>,  // strides [2, 1]
    head_layer2: Vec<ResBlock>,  // strides [2, 1]
    head_conv2: Conv2d,
    head_bn2: Option<BatchNorm>,
    // xvector chain.
    tdnn: Unit,
    block1: Vec<DenseLayer>,
    block2: Vec<DenseLayer>,
    block3: Vec<DenseLayer>,
    transit1: Unit,
    transit2: Unit,
    transit3: Unit,
    out_nl: Option<BatchNorm>, // BN-only; None = identity (absent in cstr GGUF)
    dense: Unit,               // BN affine=False
}

impl CampPlus {
    /// Load from the Fun-CosyVoice3 model dir (expects exactly one
    /// `*campplus*.gguf` inside, e.g. `cosyvoice3-campplus-f16.gguf`).
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let gguf_path = find_gguf(dir.as_ref(), "campplus")?;
        let mut reader = std::io::BufReader::new(std::fs::File::open(&gguf_path)?);
        let ct = gguf_file::Content::read(&mut reader)?;

        let mut g = GgufReader::new(&ct, &mut reader, device);

        // FCM head.
        let head_conv1 = Conv2d::new(
            g.tensor("s3.se.head.conv1.weight")?,
            g.opt_tensor("s3.se.head.conv1.bias")?,
            conv2d_cfg(1),
        );
        let head_bn1 = load_bn_opt(&mut g, "s3.se.head.bn1")?;
        let head_layer1 = vec![
            load_resblock(&mut g, "s3.se.head.layer1.0", 2)?,
            load_resblock(&mut g, "s3.se.head.layer1.1", 1)?,
        ];
        let head_layer2 = vec![
            load_resblock(&mut g, "s3.se.head.layer2.0", 2)?,
            load_resblock(&mut g, "s3.se.head.layer2.1", 1)?,
        ];
        let head_conv2 = Conv2d::new(
            g.tensor("s3.se.head.conv2.weight")?,
            g.opt_tensor("s3.se.head.conv2.bias")?,
            conv2d_cfg(1),
        );
        let head_bn2 = load_bn_opt(&mut g, "s3.se.head.bn2")?;

        // xvector units.
        let tdnn = load_unit(
            &mut g,
            "s3.se.xv.tdnn",
            Conv1dConfig {
                padding: 2,
                stride: 2,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        )?;
        let k1 = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let transit1 = load_unit(&mut g, "s3.se.xv.transit1", k1)?;
        let transit2 = load_unit(&mut g, "s3.se.xv.transit2", k1)?;
        let transit3 = load_unit(&mut g, "s3.se.xv.transit3", k1)?;
        let out_nl = load_bn_opt(&mut g, "s3.se.xv.out_nl.bn")?;
        let dense = Unit {
            conv: Conv1d::new(
                g.tensor("s3.se.xv.dense.linear.weight")?,
                g.opt_tensor("s3.se.xv.dense.linear.bias")?,
                k1,
            ),
            bn: Some(load_bn_no_affine(&mut g, "s3.se.xv.dense.nl.bn")?),
        };

        // CAMDenseTDNN blocks.
        let mut blocks: [Vec<DenseLayer>; 3] = [vec![], vec![], vec![]];
        for (bi, block) in blocks.iter_mut().enumerate() {
            let dilation = BLOCK_DILATIONS[bi];
            for li in 0..BLOCK_LAYERS[bi] {
                let p = format!("s3.se.xv.block{}.tdnnd{}", bi + 1, li + 1);
                block.push(DenseLayer {
                    nonl1_bn: load_bn(&mut g, &format!("{p}.nonl1.bn"))?,
                    l1: Conv1d::new(
                        g.tensor(&format!("{p}.l1.weight"))?,
                        g.opt_tensor(&format!("{p}.l1.bias"))?,
                        k1,
                    ),
                    nonl2_bn: load_bn_opt(&mut g, &format!("{p}.nonl2.bn"))?,
                    cam_ll: Conv1d::new(
                        g.tensor(&format!("{p}.cam.ll.weight"))?,
                        None,
                        Conv1dConfig {
                            padding: dilation,
                            stride: 1,
                            dilation,
                            groups: 1,
                            cudnn_fwd_algo: None,
                        },
                    ),
                    cam_l1: Conv1d::new(
                        g.tensor(&format!("{p}.cam.l1.weight"))?,
                        g.opt_tensor(&format!("{p}.cam.l1.bias"))?,
                        k1,
                    ),
                    cam_l2: Conv1d::new(
                        g.tensor(&format!("{p}.cam.l2.weight"))?,
                        g.opt_tensor(&format!("{p}.cam.l2.bias"))?,
                        k1,
                    ),
                });
            }
        }
        let [block1, block2, block3] = blocks;

        // Kaldi mel banks: triangles interpolated in MEL space (true kaldi /
        // torchaudio behavior) — deliberate divergence from CrispASR's
        // Hz-space approximation.
        let (mel_banks, _) = kaldi_get_mel_banks(N_MELS, PADDED, 16000.0, 20.0, 0.0, device)?;
        let mel_banks_t = mel_banks.t()?.contiguous()?;
        let povey =
            crate::utils::audio_utils::create_povey_window(WIN, candle_core::DType::F32, device)?
                .reshape((1, 1, WIN))?;

        Ok(Self {
            device: device.clone(),
            mel_banks_t,
            povey,
            head_conv1,
            head_bn1,
            head_layer1,
            head_layer2,
            head_conv2,
            head_bn2,
            tdnn,
            block1,
            block2,
            block3,
            transit1,
            transit2,
            transit3,
            out_nl,
            dense,
        })
    }

    /// Kaldi 80-bin fbank (Povey window, 25/10 ms, HTK mel, log power) with
    /// per-utterance mean subtraction -> (T, 80). Mirrors
    /// `chatterbox_campplus::compute_fbank` (no int16 scaling, dither 0).
    fn fbank(&self, wav: &Tensor) -> Result<Tensor> {
        let frames = extract_frames(wav, WIN, SHIFT)?; // (1, T, 400)
        // per-frame DC offset removal, then preemphasis 0.97.
        let frames = frames.broadcast_sub(&frames.mean_keepdim(D::Minus1)?)?;
        let shifted = pad_replicate_last_dim(&frames, (1, 0))?.affine(0.97, 0.0)?;
        let frames = frames.sub(&shifted.narrow(D::Minus1, 0, WIN)?)?;
        let frames = frames.broadcast_mul(&self.povey)?;
        let frames = frames.pad_with_zeros(D::Minus1, 0, PADDED - WIN)?;
        let power = apply_stft(&frames)?; // (1, T, 257)
        // Kaldi mel banks cover padded/2 bins (Nyquist bin dropped).
        // .contiguous(): Metal matmul rejects strided (narrowed) operands.
        let power = power.narrow(D::Minus1, 0, PADDED / 2)?.contiguous()?;
        let mel = power.broadcast_matmul(&self.mel_banks_t)?; // (1, T, 80)
        let eps = Tensor::new(1.192_092_9e-7_f32, wav.device())?.broadcast_as(mel.shape())?;
        let mel = mel.maximum(&eps)?.log()?.squeeze(0)?; // (T, 80)
        let mean = mel.mean_keepdim(0)?;
        Ok(mel.broadcast_sub(&mean)?)
    }

    /// 16 kHz mono ref wav -> 192-d speaker embedding (raw, NOT L2-normed).
    pub fn embed(&self, wav_16k: &[f32]) -> Result<Tensor> {
        if wav_16k.is_empty() {
            return Err(anyhow::anyhow!("campplus: empty input waveform"));
        }
        // extract_frames needs at least one 400-sample window.
        let mut pcm;
        let wav: &[f32] = if wav_16k.len() < WIN {
            pcm = wav_16k.to_vec();
            pcm.resize(WIN, 0.0);
            &pcm
        } else {
            wav_16k
        };
        let wav = Tensor::from_vec(wav.to_vec(), (1, wav.len()), &self.device)?;
        let feat = self.fbank(&wav)?; // (T, 80)

        // FCM head: (T, 80) -> (1, 1, 80, T) -> (1, 32, 10, T) -> (1, 320, T)
        let mut x = feat.t()?.contiguous()?.unsqueeze(0)?.unsqueeze(0)?;
        x = bn_eval_opt(&self.head_bn1, &self.head_conv1.forward(&x)?)?.relu()?;
        for b in &self.head_layer1 {
            x = b.forward(&x)?;
        }
        for b in &self.head_layer2 {
            x = b.forward(&x)?;
        }
        let y = conv2d_stride_h2(&self.head_conv2, &x)?;
        let y = bn_eval_opt(&self.head_bn2, &y)?.relu()?;
        let (_, c, h, t) = y.dims4()?;
        let x = y.contiguous()?.reshape((1, c * h, t))?;

        // xvector chain.
        let mut x = self.tdnn.conv_bn(&x)?.relu()?;
        for l in &self.block1 {
            x = l.forward(&x)?;
        }
        x = self.transit1.bn_relu_conv(&x)?;
        for l in &self.block2 {
            x = l.forward(&x)?;
        }
        x = self.transit2.bn_relu_conv(&x)?;
        for l in &self.block3 {
            x = l.forward(&x)?;
        }
        x = self.transit3.bn_relu_conv(&x)?;
        let x = bn_eval_opt(&self.out_nl, &x)?.relu()?;

        // StatsPool: concat(mean, std) over T (unbiased std, like torch).
        // Guard T == 1: unbiased var divides by (T-1) -> NaN (C++:
        // chatterbox_campplus.cpp:828-829, var = T>1 ? sumsq/(T-1) : 0).
        let mean = x.mean_keepdim(2)?;
        let std = if x.dim(2)? > 1 {
            x.var_keepdim(2)?.sqrt()?
        } else {
            mean.zeros_like()?
        };
        let stats = Tensor::cat(&[&mean, &std], 1)?; // (1, 1024, 1)

        let e = self.dense.conv_bn(&stats)?;
        let emb_dim = e.dim(1)?;
        Ok(e.reshape((emb_dim,))?)
    }

    /// 24 kHz Matcha-TTS prompt mel for the flow's `prompt_feat`: (T, 80).
    /// Delegates to the free function with this encoder's device.
    pub fn prompt_feat_24k(&self, wav_24k: &[f32]) -> Result<Tensor> {
        compute_prompt_feat_24k(wav_24k, &self.device)
    }
}

/// 24 kHz prompt mel (Matcha `mel_spectrogram` with CosyVoice defaults):
/// truncate to 10 s, manual reflect pad 720 (center=False), periodic Hann
/// 1920, hop 480, magnitude sqrt(power + 1e-9), 80-bin Slaney mel
/// (fmax 8000), natural log of clamp(mel, 1e-5). Returns (T, 80).
pub fn compute_prompt_feat_24k(wav_24k: &[f32], device: &Device) -> Result<Tensor> {
    if wav_24k.is_empty() {
        return Err(anyhow::anyhow!("prompt_feat_24k: empty input waveform"));
    }
    let n = wav_24k.len().min(PROMPT_MAX_SAMPLES);
    let mut pcm = wav_24k[..n].to_vec();
    // reflect pad of 720 needs more than 720 input samples.
    if pcm.len() <= PROMPT_PAD {
        pcm.resize(PROMPT_PAD + 1, 0.0);
    }
    let wav = Tensor::from_vec(pcm.clone(), (1, pcm.len()), device)?;
    let padded = pad_reflect_last_dim(&wav, (PROMPT_PAD, PROMPT_PAD))?;

    let window = periodic_hann(PROMPT_N_FFT, device)?.reshape((1, 1, PROMPT_N_FFT))?;
    let power = torch_stft(&padded, PROMPT_N_FFT, PROMPT_HOP, &window)?; // (1, T, 961)
    let mag = (power + 1e-9)?.sqrt()?;
    let mel_filters = mel_filter_bank(
        1 + PROMPT_N_FFT / 2,
        N_MELS,
        0.0,
        8000.0,
        PROMPT_SR as f32,
        Some("slaney"),
        MelScale::Slaney,
        false,
        device,
    )?; // (961, 80)
    let mel = mag.broadcast_matmul(&mel_filters)?;
    let mel = mel.clamp(1e-5f32, f32::INFINITY)?.log()?;
    Ok(mel.squeeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_feat_shapes() -> Result<()> {
        let device = Device::Cpu;
        // 1 s @24k -> 50 mel frames of 80 bins.
        let wav = vec![0.01f32; PROMPT_SR];
        let mel = compute_prompt_feat_24k(&wav, &device)?;
        assert_eq!(mel.dims(), &[50, N_MELS]);
        Ok(())
    }

    #[test]
    fn prompt_feat_truncates_to_10s() -> Result<()> {
        let device = Device::Cpu;
        let wav = vec![0.01f32; 12 * PROMPT_SR];
        let mel = compute_prompt_feat_24k(&wav, &device)?;
        assert_eq!(mel.dims(), &[500, N_MELS]);
        Ok(())
    }

    /// Full forward with zero weights: validates the channel/time bookkeeping
    /// of every stage (fbank -> FCM -> tdnn -> dense blocks -> stats -> dense)
    /// without needing the real GGUF.
    #[test]
    fn embed_zero_weight_shapes() -> Result<()> {
        use candle_core::DType;
        let device = Device::Cpu;
        let z = |dims: &[usize]| Tensor::zeros(dims, DType::F32, &device);
        let bn = |c: usize| -> Result<BatchNorm> {
            Ok(BatchNorm::new(
                c,
                z(&[c])?,
                z(&[c])?.affine(0.0, 1.0)?,
                z(&[c])?.affine(0.0, 1.0)?,
                z(&[c])?,
                BN_EPS,
            )?)
        };
        let bn_no_aff = |c: usize| -> Result<BatchNorm> {
            Ok(BatchNorm::new_no_bias(
                c,
                z(&[c])?,
                z(&[c])?.affine(0.0, 1.0)?,
                BN_EPS,
            )?)
        };
        let c2 = |i: usize, o: usize, k: usize| -> Result<Conv2d> {
            Ok(Conv2d::new(z(&[o, i, k, k])?, None, conv2d_cfg(1)))
        };
        let c1 = |i: usize, o: usize, k: usize, cfg: Conv1dConfig| -> Result<Conv1d> {
            Ok(Conv1d::new(z(&[o, i, k])?, None, cfg))
        };
        let rb = |stride: usize| -> Result<ResBlock> {
            Ok(ResBlock {
                conv1: c2(32, 32, 3)?,
                bn1: Some(bn(32)?),
                conv2: c2(32, 32, 3)?,
                bn2: Some(bn(32)?),
                shortcut: if stride == 2 {
                    Some((
                        Conv2d::new(
                            z(&[32, 32, 1, 1])?,
                            None,
                            Conv2dConfig {
                                padding: 0,
                                ..conv2d_cfg(1)
                            },
                        ),
                        Some(bn(32)?),
                    ))
                } else {
                    None
                },
                stride_h: stride,
            })
        };
        let dense_layer = |c_in: usize, dil: usize| -> Result<DenseLayer> {
            let k1 = Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            Ok(DenseLayer {
                nonl1_bn: bn(c_in)?,
                l1: c1(c_in, 128, 1, k1)?,
                nonl2_bn: Some(bn(128)?),
                cam_ll: c1(
                    128,
                    32,
                    3,
                    Conv1dConfig {
                        padding: dil,
                        stride: 1,
                        dilation: dil,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    },
                )?,
                cam_l1: c1(128, 64, 1, k1)?,
                cam_l2: c1(64, 32, 1, k1)?,
            })
        };
        let k1 = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        // bn_c differs per unit: tdnn BNs its conv output, transits BN their input.
        let unit = |i: usize, o: usize, k: usize, cfg: Conv1dConfig, bn_c: usize| -> Result<Unit> {
            Ok(Unit {
                conv: c1(i, o, k, cfg)?,
                bn: Some(bn(bn_c)?),
            })
        };
        let block = |n: usize, base: usize, dil: usize| -> Result<Vec<DenseLayer>> {
            (0..n).map(|i| dense_layer(base + i * 32, dil)).collect()
        };
        let (mel_banks, _) = kaldi_get_mel_banks(N_MELS, PADDED, 16000.0, 20.0, 0.0, &device)?;
        let cp = CampPlus {
            device: device.clone(),
            mel_banks_t: mel_banks.t()?.contiguous()?,
            povey: crate::utils::audio_utils::create_povey_window(WIN, DType::F32, &device)?
                .reshape((1, 1, WIN))?,
            head_conv1: c2(1, 32, 3)?,
            head_bn1: Some(bn(32)?),
            head_layer1: vec![rb(2)?, rb(1)?],
            head_layer2: vec![rb(2)?, rb(1)?],
            head_conv2: c2(32, 32, 3)?,
            head_bn2: Some(bn(32)?),
            tdnn: unit(
                320,
                128,
                5,
                Conv1dConfig {
                    padding: 2,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                128,
            )?,
            block1: block(12, 128, 1)?,
            block2: block(24, 256, 2)?,
            block3: block(16, 512, 2)?,
            transit1: unit(512, 256, 1, k1, 512)?,
            transit2: unit(1024, 512, 1, k1, 1024)?,
            transit3: unit(1024, 512, 1, k1, 1024)?,
            out_nl: Some(bn(512)?),
            dense: Unit {
                conv: c1(1024, 192, 1, k1)?,
                bn: Some(bn_no_aff(192)?),
            },
        };
        let wav: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        let emb = cp.embed(&wav)?;
        assert_eq!(emb.dims(), &[192]);
        assert!(emb.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
