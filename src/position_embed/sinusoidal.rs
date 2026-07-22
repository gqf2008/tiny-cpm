//! Ported from aha (github.com/jhqxxx/aha) src/position_embed/sinusoidal_pe.rs
use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};

use crate::position_embed::rope::compute_default_rope_parameters;

pub struct SinusoidalPositionEncoderCat {
    inv_freq: Option<Tensor>, // (1, dim / 2)
}

impl SinusoidalPositionEncoderCat {
    pub fn new(dim: Option<usize>, save_freq: bool, device: &Device) -> Result<Self> {
        let inv_freq = if save_freq && let Some(dim) = dim {
            let inv_freq = compute_default_rope_parameters(dim, 10000.0);
            let inv_freq = Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?;
            Some(inv_freq)
        } else {
            None
        };

        Ok(Self { inv_freq })
    }
    pub fn encode(
        &self,
        seqlen_offset: usize,
        seq_len: usize,
        head_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let positions = Tensor::arange(
            seqlen_offset as f32,
            (seqlen_offset + seq_len) as f32,
            device,
        )?
        .reshape((seq_len, 1))?; // (seq_len, 1)
        let inv_freq = if let Some(inv_freq) = &self.inv_freq {
            inv_freq.clone()
        } else {
            let inv_freq = compute_default_rope_parameters(head_dim, 10000.0);
            Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?
        };
        let freqs = positions.matmul(&inv_freq)?; // (seq_len, dim / 2)
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        let pos_embed = Tensor::cat(&[sin, cos], D::Minus1)?
            .contiguous()?
            .to_dtype(dtype)?;

        Ok(pos_embed)
    }
    pub fn forward(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_, seq_len, head_dim) = xs.dims3()?;
        let pos_embed = self
            .encode(seqlen_offset, seq_len, head_dim, xs.device(), xs.dtype())?
            .unsqueeze(0)?;
        let xs = xs.broadcast_add(&pos_embed)?;
        Ok(xs)
    }
}
