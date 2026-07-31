//! Fused RoPE (rotary position embedding) as a single Metal kernel, for the
//! Qwen3-TTS Q4_K talker path.
//!
//! Replaces the ~12 GPU kernels `apply_rotary_pos_emb` (src/position_embed/
//! rope.rs) issues per layer with ONE launch — the m=1 decode cost driver is
//! kernel-launch overhead (~39µs each), not the elementwise math. This does NOT
//! fork candle: candle-metal-kernels compiles MSL at runtime from a source
//! string via `Device::new_library_with_source`, and we ride candle's shared
//! command encoder, so ordering with surrounding ops is automatic.
//!
//! Fallback: on a non-Metal device (or any unexpected shape), we fall back to
//! the composite `apply_rotary_pos_emb` so the CPU test path keeps working.

use anyhow::{Result, bail};
use candle_core::metal_backend::buffer_o;
use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use candle_metal_kernels::{Output, set_params};
use objc2_metal::MTLSize;
use std::sync::OnceLock;

use crate::position_embed::rope::apply_rotary_pos_emb;

const MSL: &str = include_str!("rope_fused.metal");

// One cached pipeline per (kernel-name) — mirrors candle-metal-kernels'
// Kernels::load_pipeline but for our own source (its Source enum is closed).
static PIPELINE_F32: OnceLock<ComputePipeline> = OnceLock::new();
static PIPELINE_BF16: OnceLock<ComputePipeline> = OnceLock::new();

fn compile_kernel(device: &candle_core::MetalDevice, name: &str) -> Result<ComputePipeline> {
    let raw = device.device();
    let lib = raw
        .new_library_with_source(MSL, None)
        .map_err(|e| anyhow::anyhow!("rope_fused library: {e}"))?;
    let func = lib
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("rope_fused function {name}: {e}"))?;
    raw.new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("rope_fused pipeline {name}: {e}"))
}

fn pipeline_for(device: &candle_core::MetalDevice, dtype: DType) -> Result<&'static ComputePipeline> {
    match dtype {
        DType::F32 => Ok(PIPELINE_F32.get_or_init(|| {
            compile_kernel(device, "rope_fused_f32").expect("compile rope_fused_f32")
        })),
        DType::BF16 => Ok(PIPELINE_BF16.get_or_init(|| {
            compile_kernel(device, "rope_fused_bf16").expect("compile rope_fused_bf16")
        })),
        other => bail!("rope_fused: unsupported dtype {other:?}"),
    }
}

/// Fused RoPE for one attention layer. q `(b, q_heads, seq, head_dim)`, k
/// `(b, kv_heads, seq, head_dim)`, cos/sin broadcastable to `(1,1,seq,head_dim)`
/// (we only need the `(seq, head_dim)` values). Returns (q_rot, k_rot) with the
/// same shapes/dtypes as the inputs. Falls back to the composite op sequence on
/// any non-Metal device or shape we don't handle.
pub fn apply_rope_fused(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    // Only handle the Metal + contiguous + 4D case; anything else uses the
    // well-tested composite path.
    let device = match q.device() {
        Device::Metal(_) => q.device(),
        _ => return apply_rotary_pos_emb(q, k, cos, sin, false).map_err(Into::into),
    };
    if q.rank() != 4 || k.rank() != 4 || q.dtype() != k.dtype() {
        return apply_rotary_pos_emb(q, k, cos, sin, false).map_err(Into::into);
    }
    let dtype = q.dtype();
    if dtype != DType::F32 && dtype != DType::BF16 {
        return apply_rotary_pos_emb(q, k, cos, sin, false).map_err(Into::into);
    }

    let q_dims = q.dims4()?;
    let k_dims = k.dims4()?;
    let (b, q_heads, seq_len, head_dim) = q_dims;
    let kv_heads = k_dims.1;
    if k_dims.0 != b || k_dims.2 != seq_len || k_dims.3 != head_dim {
        return apply_rotary_pos_emb(q, k, cos, sin, false).map_err(Into::into);
    }
    // We pass a single (sb, sh, ss, sd) stride set to the kernel; q and k differ
    // in head count, so their BATCH stride differs (sb = heads*seq*hd). At b==1
    // the batch term is always 0 and the shared set is exact — that's the only
    // case the talker ever runs. For b>1, fall back to the composite path.
    if b != 1 {
        return apply_rotary_pos_emb(q, k, cos, sin, false).map_err(Into::into);
    }

    let metal_dev = device.as_metal_device()?.clone();
    let pipeline = pipeline_for(&metal_dev, dtype)?;

    // cos/sin: reduce any leading broadcast dims to a plain (seq_len, head_dim)
    // F32 buffer. At m=1 they arrive as (1,128) or unsqueezed variants.
    let cos = cos.to_dtype(DType::F32)?.reshape((seq_len, head_dim))?.contiguous()?;
    let sin = sin.to_dtype(DType::F32)?.reshape((seq_len, head_dim))?.contiguous()?;

    let q_rows = (b * q_heads * seq_len) as i64;
    let k_rows = (b * kv_heads * seq_len) as i64;
    let head_dim_i = head_dim as i64;
    let seq_len_i = seq_len as i64;

    // Input strides: q/k arrive as transpose(1,2) of the (b, seq, heads, head_dim)
    // projections — NON-contiguous when seq_len > 1. Both share the same layout
    // pattern (only head counts differ), so read the real strides from q's layout
    // and pass them to the kernel; it reconstructs each row's offset from
    // (batch, head, seq). k uses identical strides (batch/head/seq/head_dim order).
    let q_el = q.elem_count();
    let k_el = k.elem_count();

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let (sin_storage, sin_layout) = sin.storage_and_layout();

    // candle Layout::stride is in element units, dims ordered (b, heads, seq, hd).
    let st = q_layout.stride();
    if st.len() != 4 {
        return apply_rotary_pos_emb(q, k, &cos, &sin, false).map_err(Into::into);
    }
    // k must share q's stride pattern (it does: same reshape+transpose recipe),
    // else the single-stride-set assumption breaks — fall back rather than corrupt.
    if k_layout.stride() != st {
        return apply_rotary_pos_emb(q, k, &cos, &sin, false).map_err(Into::into);
    }
    let (sb, sh, ss, sd) = (st[0] as i64, st[1] as i64, st[2] as i64, st[3] as i64);

    let (q_ms, k_ms, cos_ms, sin_ms) = match (
        &*q_storage,
        &*k_storage,
        &*cos_storage,
        &*sin_storage,
    ) {
        (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c), Storage::Metal(d)) => (a, b, c, d),
        _ => bail!("rope_fused: non-metal input storage"),
    };

    let q_out_buf = metal_dev.new_buffer(q_el, dtype, "rope_q_out")?;
    let k_out_buf = metal_dev.new_buffer(k_el, dtype, "rope_k_out")?;

    {
        let guard = metal_dev.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        set_params!(
            encoder,
            (
                &buffer_o(q_ms.buffer(), q_layout, dtype),
                &buffer_o(k_ms.buffer(), k_layout, dtype),
                &buffer_o(cos_ms.buffer(), cos_layout, DType::F32),
                &buffer_o(sin_ms.buffer(), sin_layout, DType::F32),
                Output::new(&q_out_buf),
                Output::new(&k_out_buf),
                q_rows,
                k_rows,
                head_dim_i,
                seq_len_i,
                sb,
                sh,
                ss,
                sd
            )
        );
        // Grid: x = head_dim (one thread per element), y = q_rows + k_rows.
        let total_rows = (q_rows + k_rows) as usize;
        let tgs = MTLSize {
            width: head_dim.min(256),
            height: 1,
            depth: 1,
        };
        let tgc = MTLSize {
            width: head_dim.div_ceil(tgs.width),
            height: total_rows,
            depth: 1,
        };
        encoder.dispatch_thread_groups(tgc, tgs);
    }

    let q_out = Tensor::from_storage(
        Storage::Metal(MetalStorage::new(q_out_buf, metal_dev.clone(), q_el, dtype)),
        q.dims().to_vec(),
        BackpropOp::none(),
        false,
    );
    let k_out = Tensor::from_storage(
        Storage::Metal(MetalStorage::new(k_out_buf, metal_dev.clone(), k_el, dtype)),
        k.dims().to_vec(),
        BackpropOp::none(),
        false,
    );
    Ok((q_out, k_out))
}
