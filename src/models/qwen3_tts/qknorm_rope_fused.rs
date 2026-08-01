//! Fused per-head QK-RMSNorm + RoPE as a single Metal kernel, for the Qwen3-TTS
//! decode path. Replaces QKNormAttention's `q_norm`/`k_norm` (one reduction
//! kernel each) + the RoPE kernel with ONE launch — at m=1 the predictor runs 5
//! layers × 15 steps, so 3 launches/layer → 1 is a real chunk of the launch-bound
//! floor. Same no-fork runtime-MSL mechanism as `rope_fused`/`swiglu_fused`.
//!
//! Operates in the projections' `(b, seq, heads, head_dim)` layout (BEFORE the
//! transpose(1,2) that feeds SDPA): each threadgroup handles one (b, seq, head)
//! row = head_dim contiguous elements, doing the RMSNorm reduction and the rotary
//! in one pass. Output is the same layout; the caller transposes for SDPA.
//! Falls back (returns `None`) on non-Metal / wrong shape / unsupported dtype.

use anyhow::{Result, bail};
use candle_core::metal_backend::buffer_o;
use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use candle_metal_kernels::{Output, set_params};
use objc2_metal::MTLSize;
use std::sync::OnceLock;

const MSL: &str = include_str!("qknorm_rope_fused.metal");

static PIPELINE_F32: OnceLock<Option<ComputePipeline>> = OnceLock::new();
static PIPELINE_BF16: OnceLock<Option<ComputePipeline>> = OnceLock::new();

fn compile_kernel(device: &candle_core::MetalDevice, name: &str) -> Result<ComputePipeline> {
    let raw = device.device();
    let lib = raw
        .new_library_with_source(MSL, None)
        .map_err(|e| anyhow::anyhow!("qknorm_rope library: {e}"))?;
    let func = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("qknorm_rope function {name}: {e}"))?;
    raw.new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("qknorm_rope pipeline {name}: {e}"))
}

fn pipeline_for(device: &candle_core::MetalDevice, dtype: DType) -> Option<&'static ComputePipeline> {
    // Compile once and cache the OUTCOME; on any compile error cache None so a
    // model/shape we didn't expect falls back to the composite path instead of
    // panicking the whole run (QKNormAttention is shared with Qwen3-ASR, whose
    // head_dim/dtype differ from the TTS predictor).
    let cell = match dtype {
        DType::F32 => &PIPELINE_F32,
        DType::BF16 => &PIPELINE_BF16,
        _ => return None,
    };
    cell.get_or_init(|| {
        let name = match dtype {
            DType::F32 => "qknorm_rope_f32",
            _ => "qknorm_rope_bf16",
        };
        compile_kernel(device, name).ok()
    })
    .as_ref()
}

/// Fused per-head RMSNorm + RoPE. `q`/`k` are `(b, seq, {q,kv}_heads, head_dim)`
/// contiguous; `qn_w`/`kn_w` the per-head norm weights (any float dtype, cast to
/// F32); `cos`/`sin` broadcastable to `(seq, head_dim)`. Returns `Some((q, k))`
/// in the same `(b, seq, heads, head_dim)` layout, or `None` to fall back.
pub fn qknorm_rope_fused(
    q: &Tensor,
    k: &Tensor,
    qn_w: &Tensor,
    kn_w: &Tensor,
    eps: f64,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let device = match q.device() {
        Device::Metal(_) => q.device(),
        _ => return Ok(None),
    };
    if q.rank() != 4 || k.rank() != 4 || q.dtype() != k.dtype() {
        return Ok(None);
    }
    let dtype = q.dtype();
    if dtype != DType::F32 && dtype != DType::BF16 {
        return Ok(None);
    }
    let (b, seq_len, q_heads, head_dim) = q.dims4()?;
    let (kb, kseq, kv_heads, khd) = k.dims4()?;
    if kb != b || kseq != seq_len || khd != head_dim || b != 1 || head_dim > 1024 {
        return Ok(None);
    }

    let metal_dev = device.as_metal_device()?.clone();
    let pipeline = match pipeline_for(&metal_dev, dtype) {
        Some(p) => p,
        None => return Ok(None), // compile failed or unsupported dtype → composite
    };

    // Contiguous norm-layout inputs; F32 norm weights + cos/sin.
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let qn_w = qn_w.to_dtype(DType::F32)?.reshape(head_dim)?.contiguous()?;
    let kn_w = kn_w.to_dtype(DType::F32)?.reshape(head_dim)?.contiguous()?;
    let cos = cos.to_dtype(DType::F32)?.reshape((seq_len, head_dim))?.contiguous()?;
    let sin = sin.to_dtype(DType::F32)?.reshape((seq_len, head_dim))?.contiguous()?;

    let q_rows = (b * seq_len * q_heads) as i64;
    let k_rows = (b * seq_len * kv_heads) as i64;
    let head_dim_i = head_dim as i64;
    let seq_len_i = seq_len as i64;
    let eps_f = eps as f32;

    let q_el = q.elem_count();
    let k_el = k.elem_count();

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (qn_storage, qn_layout) = qn_w.storage_and_layout();
    let (kn_storage, kn_layout) = kn_w.storage_and_layout();
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let (sin_storage, sin_layout) = sin.storage_and_layout();

    let (q_ms, k_ms, qn_ms, kn_ms, cos_ms, sin_ms) = match (
        &*q_storage,
        &*k_storage,
        &*qn_storage,
        &*kn_storage,
        &*cos_storage,
        &*sin_storage,
    ) {
        (
            Storage::Metal(a),
            Storage::Metal(b),
            Storage::Metal(c),
            Storage::Metal(d),
            Storage::Metal(e),
            Storage::Metal(f),
        ) => (a, b, c, d, e, f),
        _ => bail!("qknorm_rope: non-metal input storage"),
    };

    let q_out_buf = metal_dev.new_buffer(q_el, dtype, "qknorm_q_out")?;
    let k_out_buf = metal_dev.new_buffer(k_el, dtype, "qknorm_k_out")?;

    {
        let guard = metal_dev.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        set_params!(
            encoder,
            (
                &buffer_o(q_ms.buffer(), q_layout, dtype),
                &buffer_o(k_ms.buffer(), k_layout, dtype),
                &buffer_o(qn_ms.buffer(), qn_layout, DType::F32),
                &buffer_o(kn_ms.buffer(), kn_layout, DType::F32),
                &buffer_o(cos_ms.buffer(), cos_layout, DType::F32),
                &buffer_o(sin_ms.buffer(), sin_layout, DType::F32),
                Output::new(&q_out_buf),
                Output::new(&k_out_buf),
                q_rows,
                k_rows,
                head_dim_i,
                seq_len_i,
                eps_f
            )
        );
        // One threadgroup per row, head_dim threads per group.
        let total_rows = (q_rows + k_rows) as usize;
        let tgs = MTLSize { width: head_dim.min(1024), height: 1, depth: 1 };
        let tgc = MTLSize { width: total_rows, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(tgc, tgs);
    }

    let shape_q = vec![b, seq_len, q_heads, head_dim];
    let shape_k = vec![b, seq_len, kv_heads, head_dim];
    let q_out = Tensor::from_storage(
        Storage::Metal(MetalStorage::new(q_out_buf, metal_dev.clone(), q_el, dtype)),
        shape_q,
        BackpropOp::none(),
        false,
    );
    let k_out = Tensor::from_storage(
        Storage::Metal(MetalStorage::new(k_out_buf, metal_dev.clone(), k_el, dtype)),
        shape_k,
        BackpropOp::none(),
        false,
    );
    Ok(Some((q_out, k_out)))
}
