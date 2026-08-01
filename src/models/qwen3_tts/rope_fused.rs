//! Fused RoPE (rotary position embedding) as a single Metal kernel, for the
//! Qwen3-TTS Q4_K talker path.
//!
//! Replaces the ~12 GPU kernels `apply_rotary_pos_emb` (src/position_embed/
//! rope.rs) issues per layer — per q/k: narrow×2 + affine(-1) + cat + broadcast_mul
//! ×2 + add — with a single launch. At m=1 decode this is 336 launches/frame for
//! the 28-layer talker; the fusion target is launch overhead, not math.
//!
//! The kernel consumes the post-transpose(1,2) q/k views directly (no contiguous
//! copy) and — on the cache path — appends k/v into a caller-owned preallocated
//! KV buffer, eliminating the per-step `Tensor::cat` (2 kernels/layer) from the
//! code-predictor decode path.
//!
//! Math (split-half_d / GPT-NeoX convention, matching rotate_half in rope.rs):
//!   partner(i) = i < half_d ? x[i+half_d] : x[i-half_d]      (half_d = head_dim/2)
//!   rot(i)     = i < half_d ? -partner(i) : partner(i)
//!   out[i]     = x[i]*cos[i] + rot[i]*sin[i]
//! cos/sin use the cat(freqs,freqs) layout, so cos[i]==cos[i+half_d]; we index them
//! directly (no duplication needed).
//!
//! Fallback: on a non-Metal device (or any unexpected shape), we fall back to
//! the composite `apply_rotary_pos_emb_composite` directly — NOT the public
//! `apply_rotary_pos_emb` hook, which would re-test `fusible` (still true for
//! Metal 4-D F32/BF16, e.g. b>1) and re-enter this kernel forever (stack
//! overflow). Calling the composite keeps the CPU test path working and can
//! never recurse.

use anyhow::{Result, bail};
use candle_core::metal_backend::buffer_o;
use candle_core::op::BackpropOp;
use candle_core::{DType, Device, MetalStorage, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use candle_metal_kernels::{Output, set_params};
use objc2_metal::MTLSize;
use std::sync::OnceLock;

use crate::position_embed::rope::apply_rotary_pos_emb_composite;

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

fn pipeline_for(
    device: &candle_core::MetalDevice,
    dtype: DType,
) -> Result<&'static ComputePipeline> {
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
/// `(b, kv_heads, seq, head_dim)` — the transpose(1,2) views are consumed
/// directly (no copy). Returns (q_rot, k_rot) freshly allocated contiguous.
/// Falls back to the composite op sequence on non-Metal / unsupported.
pub fn apply_rope_fused(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (q, k, _v) = rope_fused_inner(q, k, None, cos, sin, 0)?;
    Ok((q, k))
}

/// `apply_rope_fused` + **preallocated KV append**: k/v are appended directly
/// into the caller-owned KV buffers (kv_heads, kv_cap, head_dim) at position
/// `kv_pos` — eliminating the per-step `Tensor::cat` (2 kernels/layer) from the
/// code-predictor decode path. Returns (q_rot, k_view, v_view): q is a fresh
/// contiguous buffer; k/v are strided views into the KV buffers covering the
/// appended range, consumable by the fused SDPA (`call_sdpa_vector`, which takes
/// explicit strides) without a copy.
pub fn apply_rope_fused_cache(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    kv_k: &Tensor, // (kv_heads, kv_cap, head_dim)
    kv_v: &Tensor, // (kv_heads, kv_cap, head_dim)
    kv_pos: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    rope_fused_inner(q, k, Some((v, kv_k, kv_v)), cos, sin, kv_pos)
}

/// The `v` / `kv` args are `Some((v, kv_k, kv_v))` only on the cache path.
#[allow(clippy::type_complexity)]
fn rope_fused_inner(
    q: &Tensor,
    k: &Tensor,
    kv: Option<(&Tensor, &Tensor, &Tensor)>,
    cos: &Tensor,
    sin: &Tensor,
    kv_pos: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let fallback = |q: &Tensor, k: &Tensor| -> Result<(Tensor, Tensor)> {
        apply_rotary_pos_emb_composite(q, k, cos, sin, false).map_err(Into::into)
    };

    // Only handle the Metal + 4D + F32/BF16 + b==1 case; anything else uses the
    // well-tested composite path (b>1 shares one stride set in the kernel).
    let device = match q.device() {
        Device::Metal(_) => q.device(),
        _ => {
            let (q, k) = fallback(q, k)?;
            let v = match kv {
                Some((v, _, _)) => v.clone(),
                None => Tensor::zeros((), DType::F32, &Device::Cpu)?,
            };
            return Ok((q, k, v));
        }
    };
    let dtype = q.dtype();
    if dtype != k.dtype()
        || (dtype != DType::F32 && dtype != DType::BF16)
        || q.rank() != 4
        || k.rank() != 4
    {
        let (q, k) = fallback(q, k)?;
        let v = match kv {
            Some((v, _, _)) => v.clone(),
            None => Tensor::zeros((), DType::F32, &Device::Cpu)?,
        };
        return Ok((q, k, v));
    }
    let (b, q_heads, seq_len, head_dim) = q.dims4()?;
    let (kb, kv_heads, k_seq, khd) = k.dims4()?;
    if b != 1 || kb != 1 || k_seq != seq_len || khd != head_dim {
        let (q, k) = fallback(q, k)?;
        let v = match kv {
            Some((v, _, _)) => v.clone(),
            None => Tensor::zeros((), DType::F32, &Device::Cpu)?,
        };
        return Ok((q, k, v));
    }

    let metal_dev = device.as_metal_device()?.clone();
    let pipeline = pipeline_for(&metal_dev, dtype)?;

    // cos/sin: reduce any leading broadcast dims to a plain (seq_len, head_dim)
    // F32 buffer. At m=1 they arrive as (1,128) or unsqueezed variants.
    let cos = cos
        .to_dtype(DType::F32)?
        .reshape((seq_len, head_dim))?
        .contiguous()?;
    let sin = sin
        .to_dtype(DType::F32)?
        .reshape((seq_len, head_dim))?
        .contiguous()?;

    let q_rows = (b * q_heads * seq_len) as i64;
    let k_rows = (b * kv_heads * seq_len) as i64;
    let head_dim_i = head_dim as i64;
    let seq_len_i = seq_len as i64;
    let heads_q_i = q_heads as i64;
    let heads_k_i = kv_heads as i64;

    let q_el = q.elem_count();
    let k_el = k.elem_count();

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let (sin_storage, sin_layout) = sin.storage_and_layout();

    let (q_ms, k_ms, cos_ms, sin_ms) =
        match (&*q_storage, &*k_storage, &*cos_storage, &*sin_storage) {
            (Storage::Metal(a), Storage::Metal(b), Storage::Metal(c), Storage::Metal(d)) => {
                (a, b, c, d)
            }
            _ => bail!("rope_fused: non-metal input storage"),
        };

    let q_out_buf = metal_dev.new_buffer(q_el, dtype, "rope_q_out")?;
    // Output targets: fresh k/v buffers (non-cache) or the preallocated KV
    // buffers (cache). v is only read on the cache path.
    let (k_out_buf, v_out_buf) = match kv {
        Some((_, kv_k, kv_v)) => {
            let (k_storage, _) = kv_k.storage_and_layout();
            let (v_storage, _) = kv_v.storage_and_layout();
            let (a, b) = match (&*k_storage, &*v_storage) {
                (Storage::Metal(a), Storage::Metal(b)) => (a, b),
                _ => bail!("rope_fused: non-metal kv cache storage"),
            };
            (
                std::sync::Arc::new(a.buffer().clone()),
                std::sync::Arc::new(b.buffer().clone()),
            )
        }
        None => {
            let k_out = metal_dev.new_buffer(k_el, dtype, "rope_k_out")?;
            let v_out = metal_dev.new_buffer(k_el, dtype, "rope_v_out")?;
            (k_out, v_out)
        }
    };
    let (kv_cap_i, kv_pos_i, use_cache_i) = match kv {
        Some((_, kv_k, _)) => {
            let (_, cap, _) = kv_k.dims3()?;
            (cap as i64, kv_pos as i64, 1i32)
        }
        None => (seq_len_i, 0, 0i32),
    };
    // v input buffer (cache path only): pass its REAL layout — v is a different
    // narrow window of the qkv projection, so it has its own start offset (and
    // the same strides as k). The kernel indexes v with k's strides; the
    // per-tensor stride params stay identical across k and v (same shape).
    // Non-cache arm: v_out is never read (the v rows early-return), so a dummy
    // layout is fine there.
    let id_layout = candle_core::Layout::contiguous(&[1, 1, 1, 1]);
    let (v_buf, v_layout) = match kv {
        Some((v, _, _)) => {
            let (v_storage, v_layout) = v.storage_and_layout();
            match &*v_storage {
                Storage::Metal(a) => (std::sync::Arc::new(a.buffer().clone()), v_layout),
                _ => bail!("rope_fused: non-metal v storage"),
            }
        }
        None => (v_out_buf.clone(), &id_layout),
    };
    // Layout strides (elements) along the seq / head dims for q and k — the
    // kernel consumes the post-transpose(1,2) views directly (decode: head
    // stride = dim, seq stride = heads*dim) or contiguous inputs (prefill).
    let (ss_q, sh_q) = (q_layout.stride()[2], q_layout.stride()[1]);
    let (ss_k, sh_k) = (k_layout.stride()[2], k_layout.stride()[1]);

    {
        let guard = metal_dev.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(pipeline);
        set_params!(
            encoder,
            (
                &buffer_o(q_ms.buffer(), q_layout, dtype),
                &buffer_o(k_ms.buffer(), k_layout, dtype),
                &buffer_o(&v_buf, &v_layout, dtype),
                &buffer_o(cos_ms.buffer(), cos_layout, DType::F32),
                &buffer_o(sin_ms.buffer(), sin_layout, DType::F32),
                Output::new(&q_out_buf),
                Output::new(&k_out_buf),
                Output::new(&v_out_buf),
                q_rows,
                k_rows,
                head_dim_i,
                seq_len_i,
                heads_q_i,
                heads_k_i,
                ss_q as i64,
                sh_q as i64,
                ss_k as i64,
                sh_k as i64,
                kv_cap_i,
                kv_pos_i,
                use_cache_i
            )
        );
        // Grid: x = head_dim (one thread per element), y = q_rows + k_rows + v_rows.
        let total_rows = (q_rows + k_rows + k_rows) as usize;
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
    match kv {
        Some((_, kv_k, kv_v)) => {
            // k/v views over the appended range: (1, kv_heads, kv_pos+seq_len, head_dim)
            // via unsqueeze(0) (a view, no copy) — the fused SDPA takes explicit
            // strides and reads the strided views directly. Per-head stride is
            // kv_cap*head_dim (the preallocated buffers' dim-1 stride).
            let kv_k = kv_k.narrow(1, 0, kv_pos + seq_len)?.unsqueeze(0)?;
            let kv_v = kv_v.narrow(1, 0, kv_pos + seq_len)?.unsqueeze(0)?;
            Ok((q_out, kv_k, kv_v))
        }
        None => {
            let k_out = Tensor::from_storage(
                Storage::Metal(MetalStorage::new(k_out_buf, metal_dev.clone(), k_el, dtype)),
                k.dims().to_vec(),
                BackpropOp::none(),
                false,
            );
            let v_out = Tensor::from_storage(
                Storage::Metal(MetalStorage::new(v_out_buf, metal_dev.clone(), k_el, dtype)),
                k.dims().to_vec(),
                BackpropOp::none(),
                false,
            );
            Ok((q_out, k_out, v_out))
        }
    }
}
