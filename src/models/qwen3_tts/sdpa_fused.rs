//! Fused single-query attention (flash-decode style) via candle-metal-kernels'
//! built-in `call_sdpa_vector` — ONE Metal kernel replacing the eager attention
//! tail (`repeat_kv` ×2 + `contiguous` ×3 + QK^T matmul + scale-mul + softmax +
//! attn·V matmul ≈ 9 kernels) per layer. The code predictor runs 15 sequential
//! 5-layer steps per frame and is launch-bound at m=1, so collapsing the
//! attention tail is the biggest remaining per-frame lever. No fork: we reuse
//! `MetalDevice::kernels()` (candle's own pipeline cache) and its shared
//! `command_encoder` for ordering.
//!
//! `call_sdpa_vector` requires: q `(b, q_heads, q_seq=1, head_dim)`, k/v
//! `(b, kv_heads, kv_seq, head_dim)`, all contiguous, head_dim ∈ {32,64,96,128,…},
//! GQA via `gqa_factor = q_heads / kv_heads`, NO mask (decode step has none).
//! Anything not matching falls back to the eager path (returns `None`).

use anyhow::{Result, bail};
use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Storage, Tensor};
use candle_metal_kernels::kernels::sdpa::{SdpaDType, call_sdpa_vector};

/// Fused decode-step attention: softmax(q·K^T · scale)·V for a single query
/// position. `q`/`k`/`v` are the post-RoPE, post-KV-cache tensors; `scale` is
/// `1/sqrt(head_dim)`. Returns `Some(out)` shaped `(b, q_heads, 1, head_dim)` on
/// success, `None` when the inputs aren't the contiguous Metal decode shape the
/// kernel needs (caller falls back to `eager_attention_forward`).
pub fn sdpa_vector_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    // Only the Metal decode path: q_seq == 1, 4-D, matching dtypes, F32/BF16.
    let device = match q.device() {
        Device::Metal(_) => q.device(),
        _ => return Ok(None),
    };
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Ok(None);
    }
    let dtype = q.dtype();
    if dtype != k.dtype() || dtype != v.dtype() {
        return Ok(None);
    }
    let itype = match dtype {
        DType::F32 => SdpaDType::F32,
        DType::BF16 => SdpaDType::BF16,
        DType::F16 => SdpaDType::F16,
        _ => return Ok(None),
    };
    let (b, q_heads, q_seq, head_dim) = q.dims4()?;
    let (kb, kv_heads, kv_seq, khd) = k.dims4()?;
    let (vb, vv_heads, vv_seq, vhd) = v.dims4()?;
    if q_seq != 1
        || kb != b
        || vb != b
        || khd != head_dim
        || vhd != head_dim
        || vv_seq != kv_seq
        || vv_heads != kv_heads
        || q_heads % kv_heads != 0
        || !matches!(head_dim, 32 | 64 | 96 | 128 | 256 | 512)
    {
        return Ok(None);
    }

    // The kernel reads raw buffers with explicit strides; it needs contiguous q
    // and contiguous k/v in (b, heads, seq, head_dim) layout.
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    let metal_dev = device.as_metal_device()?.clone();
    let out_el = b * q_heads * q_seq * head_dim;
    let out_buf = metal_dev.new_buffer(out_el, dtype, "sdpa_vec_out")?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (q_ms, k_ms, v_ms) = match (&*q_storage, &*k_storage, &*v_storage) {
        (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c)) => (a, b, c),
        _ => bail!("sdpa_vector: non-metal input storage"),
    };

    // call_sdpa_vector wants byte offsets + per-head (dim-1) strides for k/v.
    let k_stride = k_layout.stride();
    let v_stride = v_layout.stride();
    let q_off = q_layout.start_offset() * dtype.size_in_bytes();
    let k_off = k_layout.start_offset() * dtype.size_in_bytes();
    let v_off = v_layout.start_offset() * dtype.size_in_bytes();

    {
        let guard = metal_dev.command_encoder()?;
        call_sdpa_vector(
            metal_dev.device(),
            &guard,
            metal_dev.kernels(),
            q_off,
            q.dims(),
            q_ms.buffer(),
            k_off,
            k.dims(),
            k_stride,
            k_ms.buffer(),
            v_off,
            v_stride,
            v_ms.buffer(),
            &out_buf,
            scale,
            1.0, // softcapping: 1.0 = disabled
            itype,
        )
        .map_err(|e| anyhow::anyhow!("sdpa_vector: {e}"))?;
    }

    Ok(Some(Tensor::from_storage(
        Storage::Metal(MetalStorage::new(out_buf, metal_dev.clone(), out_el, dtype)),
        vec![b, q_heads, q_seq, head_dim],
        BackpropOp::none(),
        false,
    )))
}
