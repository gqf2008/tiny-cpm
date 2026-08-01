//! Fused RoPE + V-copy as a single Metal kernel, for the Qwen3-TTS Q4_K talker
//! and code-predictor paths.
//!
//! Replaces the ~12 GPU kernels `apply_rotary_pos_emb` (src/position_embed/
//! rope.rs) issues per layer — per q/k: narrow×2 + affine(-1) + cat + broadcast_mul
//! ×2 + add — with a single launch. At m=1 decode this is 336 launches/frame for
//! the 28-layer talker; the fusion target is launch overhead, not math.
//!
//! The kernel consumes the **post-transpose(1,2) views directly** (no contiguous
//! copy — saves 2 kernels/layer): q/k/v arrive as (b, heads, seq, head_dim) with
//! stride (heads*dim, 1, dim, 1), so the only per-tensor difference is the batch
//! stride (= heads*dim). It emits:
//!   - q → fresh contiguous (b, q_heads, seq, head_dim) buffer
//!   - k/v → either fresh contiguous buffers (talker path) or appended into a
//!     caller-owned preallocated KV buffer (code-predictor path, eliminating the
//!     per-step `Tensor::cat` ×2/layer)
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
#include <metal_stdlib>
using namespace metal;

template <typename T>
static void rope_fused_impl(
    device const T* q [[buffer(0)]],
    device const T* k [[buffer(1)]],
    device const T* v [[buffer(2)]],
    device const float* cos [[buffer(3)]],   // (seq_len, head_dim) F32
    device const float* sin [[buffer(4)]],   // (seq_len, head_dim) F32
    device T* q_out [[buffer(5)]],           // fresh (b, q_heads, seq, head_dim)
    device T* kv_k [[buffer(6)]],            // cache: (kv_heads, kv_cap, dim); else fresh k_out
    device T* kv_v [[buffer(7)]],            // cache: (kv_heads, kv_cap, dim); else fresh v_out
    constant int64_t& q_rows [[buffer(8)]],  // b * q_heads * seq_len
    constant int64_t& k_rows [[buffer(9)]],  // b * kv_heads * seq_len
    constant int64_t& head_dim [[buffer(10)]],
    constant int64_t& seq_len [[buffer(11)]],
    constant int64_t& heads_q [[buffer(12)]], // q_heads (row decomposition)
    constant int64_t& heads_k [[buffer(13)]], // kv_heads (row decomposition)
    constant int64_t& ss_q [[buffer(14)]],    // q stride along the seq dim (elems)
    constant int64_t& sh_q [[buffer(15)]],    // q stride along the head dim (elems)
    constant int64_t& ss_k [[buffer(16)]],    // k/v stride along the seq dim (elems)
    constant int64_t& sh_k [[buffer(17)]],    // k/v stride along the head dim (elems)
    constant int64_t& kv_cap [[buffer(18)]],  // per-head stride in kv_k/kv_v (cache mode)
    constant int64_t& kv_pos [[buffer(19)]],  // append position within each head (cache mode)
    constant int32_t& use_cache [[buffer(20)]],
    uint2 gid [[thread_position_in_grid]]) {
    const int64_t row = gid.y;
    const int i = gid.x;
    if (i >= head_dim) return;

    const int half_d = (int)(head_dim / 2);
    // cos/sin are indexed by the sequence position.
    const int64_t cs_off = (seq_len > 0 ? (row % seq_len) : 0) * head_dim + i;
    const float c = cos[cs_off];
    const float s = sin[cs_off];

    // Select source by row: q rows first, then k rows, then v rows. Strides are
    // passed in from the layouts — the transpose(1,2) views used at decode have
    // (seq: heads*dim, head: dim) while contiguous prefill inputs have
    // (seq: dim, head: seq_len*dim); the params keep both correct.
    device const T* inp;
    int64_t r;       // row within the selected tensor
    int64_t heads;   // heads-per-batch for that tensor
    int64_t ss;      // seq-dim stride for that tensor
    int64_t sh;      // head-dim stride for that tensor
    bool is_rot;     // q/k get RoPE, v is a raw copy
    bool is_q;       // q goes to a fresh contiguous buffer
    if (row < q_rows) {
        inp = q; r = row; is_rot = true; is_q = true; heads = heads_q; ss = ss_q; sh = sh_q;
    } else if (row < q_rows + k_rows) {
        inp = k; r = row - q_rows; is_rot = true; is_q = false; heads = heads_k; ss = ss_k; sh = sh_k;
    } else {
        inp = v; r = row - q_rows - k_rows; is_rot = false; is_q = false; heads = heads_k; ss = ss_k; sh = sh_k;
        // v is only consumed on the cache path; without it, skip (no read).
        if (!use_cache) return;
    }
    // r spans (batch, head, seq) row-major (view layout): batch = rem/heads,
    // head = rem%heads, seq = r % seq_len. The batch term is always 0 (b==1).
    const int64_t rem = seq_len > 0 ? (r / seq_len) : 0;
    const int64_t seq = seq_len > 0 ? (r % seq_len) : 0;
    const int64_t head = heads > 0 ? (rem % heads) : 0;

    // Element offset in the source: head*sh + seq*ss + i (b==1, sd==1).
    const int64_t base = head * sh + seq * ss;
    const float x = (float)inp[base + i];
    const float rot = is_rot ? (i < half_d ? -(float)inp[base + i + half_d]
                                           : (float)inp[base + i - half_d])
                             : 0.f;
    const T val = is_rot ? (T)(x * c + rot * s) : (T)x;

    if (is_q) {
        // q → fresh contiguous (b, q_heads, seq, head_dim) → row*head_dim.
        q_out[r * head_dim + i] = val;
    } else if (use_cache) {
        // k/v → append into the preallocated KV buffer at (head, kv_pos + seq).
        if (is_rot) {
            kv_k[(head * kv_cap + kv_pos + seq) * head_dim + i] = val;
        } else {
            kv_v[(head * kv_cap + kv_pos + seq) * head_dim + i] = val;
        }
    } else {
        // Fresh contiguous k_out/v_out (b, kv_heads, seq, head_dim) → row*head_dim.
        if (is_rot) {
            kv_k[r * head_dim + i] = val;
        } else {
            kv_v[r * head_dim + i] = val;
        }
    }
}

[[host_name("rope_fused_f32")]]
kernel void rope_fused_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* cos [[buffer(3)]],
    device const float* sin [[buffer(4)]],
    device float* q_out [[buffer(5)]],
    device float* kv_k [[buffer(6)]],
    device float* kv_v [[buffer(7)]],
    constant int64_t& q_rows [[buffer(8)]],
    constant int64_t& k_rows [[buffer(9)]],
    constant int64_t& head_dim [[buffer(10)]],
    constant int64_t& seq_len [[buffer(11)]],
    constant int64_t& heads_q [[buffer(12)]],
    constant int64_t& heads_k [[buffer(13)]],
    constant int64_t& ss_q [[buffer(14)]],
    constant int64_t& sh_q [[buffer(15)]],
    constant int64_t& ss_k [[buffer(16)]],
    constant int64_t& sh_k [[buffer(17)]],
    constant int64_t& kv_cap [[buffer(18)]],
    constant int64_t& kv_pos [[buffer(19)]],
    constant int32_t& use_cache [[buffer(20)]],
    uint2 gid [[thread_position_in_grid]]) {
    rope_fused_impl(q, k, v, cos, sin, q_out, kv_k, kv_v, q_rows, k_rows, head_dim, seq_len, heads_q, heads_k, ss_q, sh_q, ss_k, sh_k, kv_cap, kv_pos, use_cache, gid);
}

[[host_name("rope_fused_bf16")]]
kernel void rope_fused_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const float* cos [[buffer(3)]],
    device const float* sin [[buffer(4)]],
    device bfloat* q_out [[buffer(5)]],
    device bfloat* kv_k [[buffer(6)]],
    device bfloat* kv_v [[buffer(7)]],
    constant int64_t& q_rows [[buffer(8)]],
    constant int64_t& k_rows [[buffer(9)]],
    constant int64_t& head_dim [[buffer(10)]],
    constant int64_t& seq_len [[buffer(11)]],
    constant int64_t& heads_q [[buffer(12)]],
    constant int64_t& heads_k [[buffer(13)]],
    constant int64_t& ss_q [[buffer(14)]],
    constant int64_t& sh_q [[buffer(15)]],
    constant int64_t& ss_k [[buffer(16)]],
    constant int64_t& sh_k [[buffer(17)]],
    constant int64_t& kv_cap [[buffer(18)]],
    constant int64_t& kv_pos [[buffer(19)]],
    constant int32_t& use_cache [[buffer(20)]],
    uint2 gid [[thread_position_in_grid]]) {
    rope_fused_impl(q, k, v, cos, sin, q_out, kv_k, kv_v, q_rows, k_rows, head_dim, seq_len, heads_q, heads_k, ss_q, sh_q, ss_k, sh_k, kv_cap, kv_pos, use_cache, gid);
}
