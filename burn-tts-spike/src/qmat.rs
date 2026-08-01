//! Q4_K quantized GEMV for the talker's heavy 2-D weights — a custom cubecl
//! kernel that dequantizes inline (no dequant-to-float-then-matmul path).
//!
//! Weight layout: row-major (n, k) f32 → per 32-elem block: f32 min + f32
//! scale=(max-min)/15 + 4 u32 packing 32 4-bit values (low nibble first).
//! Memory: 0.5 B/elem values + 0.25 B/elem params ≈ 0.75 B/elem vs 2 B f16
//! (and 4 B f32) — the m=1 decode bandwidth win that makes Q4_K sub-realtime
//! on candle. Runs entirely in F32 like candle's QMatMul (Metal kernels take
//! F32 activations).

use anyhow::{Result, anyhow};
use burn::tensor::{DType, Tensor};
use burn_backend::Shape;
use burn_tensor::BackendPrimitive;
use burn_wgpu::{CubeTensor, Metal, WgpuRuntime};
use cubecl::prelude::*;
use cubecl::wgpu::MslCompiler;

/// The concrete cubecl runtime burn-wgpu's `Metal` backend uses (MslCompiler).
pub type RT = WgpuRuntime<MslCompiler>;

/// Anchor carrying the compute client (any tensor on the target device works).
pub struct RTAnchor(pub CubeTensor<RT>);

#[cube(launch_unchecked)]
fn q4k_gemv<F: Float>(
    x: &[F],        // (K,)
    values: &[u32], // (N, 4*blocks) — per-row contiguous, 4 u32 per 32-elem block
    scales: &[F],   // (N, blocks)
    mins: &[F],     // (N, blocks)
    out: &mut [F],  // (N,)
    blocks: u32,
) {
    // All index math in u32 (cube range vars are u32; ABSOLUTE_POS is usize).
    let n = ABSOLUTE_POS as u32;
    if n < out.len() as u32 {
        let mut acc = F::new(0.0);
        for b in 0..blocks {
            let row = n * blocks + b;
            let s = scales[row as usize];
            let m = mins[row as usize];
            for u in 0..4 {
                let packed = values[(n * blocks * 4 + b * 4 + u) as usize];
                for i in 0..8 {
                    let q = (packed >> (i * 4)) & 0xF_u32;
                    let v = m + F::cast_from(q) * s;
                    acc += v * x[(b * 32 + u * 8 + i) as usize];
                }
            }
        }
        out[n as usize] = acc;
    }
}

/// Quantize a row-major (n, k) f32 weight to Q4_K blocks (see module docs).
pub fn quantize_q4k(w: &[f32], k: usize, n: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let blocks = k / 32;
    let mut values = vec![0u32; n * blocks * 4];
    let mut scales = vec![0f32; n * blocks];
    let mut mins = vec![0f32; n * blocks];
    for row in 0..n {
        for b in 0..blocks {
            let base = row * k + b * 32;
            let slice = &w[base..base + 32];
            let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if (max - min) > 0.0 {
                (max - min) / 15.0
            } else {
                1.0
            };
            mins[row * blocks + b] = min;
            scales[row * blocks + b] = scale;
            for u in 0..4 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let q = ((slice[u * 8 + i] - min) / scale).round().clamp(0.0, 15.0) as u32;
                    packed |= q << (i * 4);
                }
                values[row * blocks * 4 + b * 4 + u] = packed;
            }
        }
    }
    (values, scales, mins)
}

/// One quantized (K→N) projection. Input (1, 1, K) f16/F32 → (1, 1, N) F32.
pub struct Q4KMatmul {
    values: CubeTensor<RT>,
    scales: CubeTensor<RT>,
    mins: CubeTensor<RT>,
    k: usize,
    n: usize,
    blocks: usize,
    device: burn::tensor::Device,
}

impl Q4KMatmul {
    /// Quantize `w` (row-major (n, k) f32) and upload. `anchor` supplies the
    /// compute client (any tensor on the target device works).
    pub fn new(w: &[f32], k: usize, n: usize, anchor: &CubeTensor<RT>) -> Result<Self> {
        let (values, scales, mins) = quantize_q4k(w, k, n);
        let client = anchor.client.clone();
        let device = anchor.device.clone();
        let mk = |data: Vec<u32>| {
            CubeTensor::new_contiguous(
                client.clone(),
                device.clone(),
                Shape::new([data.len()]),
                client.create_from_slice(u32::as_bytes(&data)),
                DType::U32,
            )
        };
        let mf = |data: Vec<f32>| {
            CubeTensor::new_contiguous(
                client.clone(),
                device.clone(),
                Shape::new([data.len()]),
                client.create_from_slice(f32::as_bytes(&data)),
                DType::F32,
            )
        };
        let n_elem = n * (k / 32);
        Ok(Self {
            values: mk(values),
            scales: mf(scales),
            mins: mf(mins),
            k,
            n,
            blocks: k / 32,
            device: burn::tensor::Device::new(device.clone()),
        })
    }

    /// Debug: same GEMV with a raw CPU slice (no burn tensor path) — used to
    /// bisect burn-extraction vs kernel issues.
    pub fn raw_forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let xh = self.values.client.create_from_slice(f32::as_bytes(x));
        let out = self.values.client.empty(self.n * 4);
        let dim = CubeDim::new_1d(256);
        let count = CubeCount::Static(((self.n + 255) / 256) as u32, 1, 1);
        unsafe {
            q4k_gemv::launch_unchecked::<f32, RT>(
                &self.values.client,
                count,
                dim,
                BufferArg::from_raw_parts(xh, self.k),
                BufferArg::from_raw_parts(self.values.handle.clone(), self.n * self.blocks * 4),
                BufferArg::from_raw_parts(self.scales.handle.clone(), self.n * self.blocks),
                BufferArg::from_raw_parts(self.mins.handle.clone(), self.n * self.blocks),
                BufferArg::from_raw_parts(out.clone(), self.n),
                self.blocks as u32,
            )
        };
        self.values.client.sync();
        let bytes = self.values.client.read_one(out).unwrap();
        Ok(f32::from_bytes(&bytes).to_vec())
    }

    /// `x`: (1, S, K) in f16 or f32 → (1, S, N) F32. S>1 (prefill) runs one
    /// GEMV per token (the m=1 kernel is per-output-row; a batched variant is a
    /// later optimization).
    pub fn forward(
        &self,
        x: Tensor<3, burn::tensor::Float>,
    ) -> Result<Tensor<3, burn::tensor::Float>> {
        let [b, s, k] = x.dims();
        if b != 1 || k != self.k {
            return Err(anyhow!(
                "Q4KMatmul: expected (1, S, K={}) got ({b}, {s}, {k})",
                self.k
            ));
        }
        let xf = x.cast(DType::F32);
        if s == 1 {
            return self.forward_single(xf);
        }
        let mut outs = Vec::with_capacity(s);
        for i in 0..s {
            outs.push(self.forward_single(xf.clone().narrow(1, i, 1))?);
        }
        Ok(Tensor::cat(outs, 1))
    }

    /// Single-token GEMV: (1, 1, K) f32 → (1, 1, N) f32.
    fn forward_single(
        &self,
        x: Tensor<3, burn::tensor::Float>,
    ) -> Result<Tensor<3, burn::tensor::Float>> {
        let x = x.reshape([1, self.k]);
        let xc = x
            .try_into_primitive::<Metal>()
            .map_err(|e| anyhow!("x into primitive: {e:?}"))?;
        use burn_tensor::BackendPrimitive as _;

        let out = self.values.client.empty(self.n * 4);
        let dim = CubeDim::new_1d(256);
        let count = CubeCount::Static(((self.n + 255) / 256) as u32, 1, 1);
        unsafe {
            q4k_gemv::launch_unchecked::<f32, RT>(
                &self.values.client,
                count,
                dim,
                BufferArg::from_raw_parts(xc.handle.clone(), self.k),
                BufferArg::from_raw_parts(self.values.handle.clone(), self.n * self.blocks * 4),
                BufferArg::from_raw_parts(self.scales.handle.clone(), self.n * self.blocks),
                BufferArg::from_raw_parts(self.mins.handle.clone(), self.n * self.blocks),
                BufferArg::from_raw_parts(out.clone(), self.n),
                self.blocks as u32,
            )
        };
        let out_c = CubeTensor::new_contiguous(
            self.values.client.clone(),
            self.values.device.clone(),
            Shape::new([self.n]),
            out,
            DType::F32,
        );
        let out_t: Tensor<1, burn::tensor::Float> = Tensor::from_primitive::<Metal>(out_c);
        Ok(out_t.reshape([1, 1, self.n]))
    }
}
