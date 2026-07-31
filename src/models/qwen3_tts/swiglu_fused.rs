//! Fused SwiGLU (`silu(gate) * up`) as a single Metal kernel — the candle
//! equivalent of mlx-audio's `@mx.compile def swiglu` (talker.py:317), which is
//! one of the two elementwise ops mlx explicitly compiles (the other is RoPE,
//! already fused in `rope_fused.rs`). Same no-fork mechanism: candle-metal-kernels
//! compiles the MSL at runtime and we ride candle's shared command encoder.
//!
//! Fallback: on a non-Metal device we return `None` and the caller uses the
//! composite `silu(gate) * up` ops, so the CPU path keeps working.

use anyhow::{Result, bail};
use candle_core::metal_backend::buffer_o;
use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use candle_metal_kernels::{Output, set_params};
use objc2_metal::MTLSize;
use std::sync::OnceLock;

const MSL: &str = include_str!("swiglu_fused.metal");

static PIPELINE_F32: OnceLock<ComputePipeline> = OnceLock::new();
static PIPELINE_BF16: OnceLock<ComputePipeline> = OnceLock::new();

fn compile_kernel(device: &candle_core::MetalDevice, name: &str) -> Result<ComputePipeline> {
    let raw = device.device();
    let lib = raw
        .new_library_with_source(MSL, None)
        .map_err(|e| anyhow::anyhow!("swiglu_fused library: {e}"))?;
    let func = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("swiglu_fused function {name}: {e}"))?;
    raw.new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("swiglu_fused pipeline {name}: {e}"))
}

fn pipeline_for(device: &candle_core::MetalDevice, dtype: DType) -> Result<&'static ComputePipeline> {
    match dtype {
        DType::F32 => Ok(PIPELINE_F32.get_or_init(|| {
            compile_kernel(device, "swiglu_fused_f32").expect("compile swiglu_fused_f32")
        })),
        DType::BF16 => Ok(PIPELINE_BF16.get_or_init(|| {
            compile_kernel(device, "swiglu_fused_bf16").expect("compile swiglu_fused_bf16")
        })),
        other => bail!("swiglu_fused: unsupported dtype {other:?}"),
    }
}

/// Fused `silu(gate) * up` for one MLP. Both inputs must share a shape/dtype and
/// be on Metal; returns `None` otherwise (caller falls back to the composite ops).
/// Output has the same shape/dtype as the inputs.
pub fn swiglu_fused(gate: &Tensor, up: &Tensor) -> Option<Result<Tensor>> {
    if gate.shape() != up.shape() || gate.dtype() != up.dtype() {
        return None;
    }
    let dtype = gate.dtype();
    if dtype != DType::F32 && dtype != DType::BF16 {
        return None;
    }
    let device = match gate.device() {
        Device::Metal(_) => gate.device(),
        _ => return None,
    };
    Some(swiglu_fused_inner(gate, up, device, dtype))
}

fn swiglu_fused_inner(gate: &Tensor, up: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
    let metal_dev = device.as_metal_device()?.clone();
    let pipeline = pipeline_for(&metal_dev, dtype)?;

    // Flatten to a contiguous 1-D element stream (the kernel is index-only).
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let numel = gate.elem_count();
    let numel_i = numel as i64;

    let (g_storage, g_layout) = gate.storage_and_layout();
    let (u_storage, u_layout) = up.storage_and_layout();
    let (g_ms, u_ms) = match (&*g_storage, &*u_storage) {
        (Storage::Metal(a), Storage::Metal(b)) => (a, b),
        _ => bail!("swiglu_fused: non-metal input storage"),
    };

    let out_buf = metal_dev.new_buffer(numel, dtype, "swiglu_out")?;

    {
        let guard = metal_dev.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        set_params!(
            encoder,
            (
                &buffer_o(g_ms.buffer(), g_layout, dtype),
                &buffer_o(u_ms.buffer(), u_layout, dtype),
                Output::new(&out_buf),
                numel_i
            )
        );
        let tgs = MTLSize { width: numel.min(256), height: 1, depth: 1 };
        let tgc = MTLSize { width: numel.div_ceil(tgs.width), height: 1, depth: 1 };
        encoder.dispatch_thread_groups(tgc, tgs);
    }

    Ok(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(out_buf, metal_dev.clone(), numel, dtype)),
        gate.dims().to_vec(),
        BackpropOp::none(),
        false,
    ))
}
