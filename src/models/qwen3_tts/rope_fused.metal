// Fused RoPE (rotary position embedding) for the Qwen3-TTS Q4_K talker path.
//
// Replaces the ~12 GPU kernels that `apply_rotary_pos_emb` (src/position_embed/
// rope.rs) issues per layer — per q/k: narrow×2 + affine(-1) + cat + broadcast_mul
// ×2 + add — with a single launch. At m=1 decode this is 336 launches/frame for
// the 28-layer talker; the fusion target is launch overhead, not math.
//
// Math (split-half_d / GPT-NeoX convention, matching rotate_half in rope.rs):
//   partner(i) = i < half_d ? x[i+half_d] : x[i-half_d]      (half_d = head_dim/2)
//   rot(i)     = i < half_d ? -partner(i) : partner(i)
//   out[i]     = x[i]*cos[i] + rot[i]*sin[i]
// cos/sin use the cat(freqs,freqs) layout, so cos[i]==cos[i+half_d]; we index them
// directly (no duplication needed). At m=1 cos/sin are a single (head_dim,) row.
//
// One launch covers q AND k: rows [0, q_rows) read q, rows [q_rows, q_rows+k_rows)
// read k. Each row is `head_dim` elements; cos/sin are indexed by the row's
// sequence position (row % seq_len for prefill; at m=1 every row is position 0).
//
// The q/k inputs are the `transpose(1,2)` of the (b, seq, heads, head_dim)
// projections, i.e. NON-contiguous when seq_len > 1 (their storage order is
// seq-major, head-minor). The composite candle path handles this via strided
// views; a raw-elementcount kernel that assumed `base = row*head_dim` read the
// WRONG elements on the prefill pass and poisoned the whole KV cache. So we take
// the real input strides and compute each row's offset from (b, head, seq); the
// OUTPUT is written contiguous (row*head_dim) so downstream consumers see a
// standard (b, heads, seq, head_dim) tensor.
#include <metal_stdlib>
using namespace metal;

template <typename T>
static void rope_fused_impl(
    device const T* q [[buffer(0)]],
    device const T* k [[buffer(1)]],
    device const float* cos [[buffer(2)]],   // (seq_len, head_dim) F32
    device const float* sin [[buffer(3)]],   // (seq_len, head_dim) F32
    device T* q_out [[buffer(4)]],
    device T* k_out [[buffer(5)]],
    constant int64_t& q_rows [[buffer(6)]],  // batch * q_heads * seq_len
    constant int64_t& k_rows [[buffer(7)]],  // batch * kv_heads * seq_len
    constant int64_t& head_dim [[buffer(8)]],
    constant int64_t& seq_len [[buffer(9)]],
    constant int64_t& sb [[buffer(10)]],     // input stride, batch dim
    constant int64_t& sh [[buffer(11)]],     // input stride, head dim
    constant int64_t& ss [[buffer(12)]],     // input stride, seq dim
    constant int64_t& sd [[buffer(13)]],     // input stride, head_dim dim
    uint2 gid [[thread_position_in_grid]]) {
    const int64_t row = gid.y;
    const int i = gid.x;
    if (i >= head_dim) return;

    const int half_d = (int)(head_dim / 2);
    // cos/sin are indexed by the sequence position.
    const int64_t cs_off = (seq_len > 0 ? (row % seq_len) : 0) * head_dim + i;
    const float c = cos[cs_off];
    const float s = sin[cs_off];

    // Select source/dest buffer by row (q rows first, then k rows) and get the
    // heads-per-batch for that tensor so we can split the row into (batch, head).
    device const T* inp;
    device T* outp;
    int64_t r;       // row within the selected tensor
    int64_t heads;   // heads-per-batch for that tensor
    if (row < q_rows) {
        inp = q; outp = q_out; r = row;
        heads = q_rows / (seq_len > 0 ? seq_len : 1);
        heads = heads > 0 ? heads : 1;
    } else if (row < q_rows + k_rows) {
        inp = k; outp = k_out; r = row - q_rows;
        heads = k_rows / (seq_len > 0 ? seq_len : 1);
        heads = heads > 0 ? heads : 1;
    } else {
        return;
    }
    // r spans (batch, head, seq) row-major: batch = rem/heads, head = rem%heads.
    const int64_t rem = seq_len > 0 ? (r / seq_len) : 0;
    const int64_t seq = seq_len > 0 ? (r % seq_len) : 0;
    const int64_t batch = heads > 0 ? (rem / heads) : 0;
    const int64_t head = heads > 0 ? (rem % heads) : 0;

    // Real strided input offset (handles the transpose(1,2) non-contiguity).
    const int64_t partner_i = i < half_d ? i + half_d : i - half_d;
    const int64_t base = batch * sb + head * sh + seq * ss;
    const float x = (float)inp[base + i * sd];
    const float partner = (float)inp[base + partner_i * sd];
    const float rot = i < half_d ? -partner : partner;

    // Output is contiguous (b, heads, seq, head_dim) → row*head_dim.
    outp[r * head_dim + i] = (T)(x * c + rot * s);
}

[[host_name("rope_fused_f32")]]
kernel void rope_fused_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* cos [[buffer(2)]],
    device const float* sin [[buffer(3)]],
    device float* q_out [[buffer(4)]],
    device float* k_out [[buffer(5)]],
    constant int64_t& q_rows [[buffer(6)]],
    constant int64_t& k_rows [[buffer(7)]],
    constant int64_t& head_dim [[buffer(8)]],
    constant int64_t& seq_len [[buffer(9)]],
    constant int64_t& sb [[buffer(10)]],
    constant int64_t& sh [[buffer(11)]],
    constant int64_t& ss [[buffer(12)]],
    constant int64_t& sd [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]]) {
    rope_fused_impl(q, k, cos, sin, q_out, k_out, q_rows, k_rows, head_dim, seq_len, sb, sh, ss, sd, gid);
}

[[host_name("rope_fused_bf16")]]
kernel void rope_fused_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const float* cos [[buffer(2)]],
    device const float* sin [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    constant int64_t& q_rows [[buffer(6)]],
    constant int64_t& k_rows [[buffer(7)]],
    constant int64_t& head_dim [[buffer(8)]],
    constant int64_t& seq_len [[buffer(9)]],
    constant int64_t& sb [[buffer(10)]],
    constant int64_t& sh [[buffer(11)]],
    constant int64_t& ss [[buffer(12)]],
    constant int64_t& sd [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]]) {
    rope_fused_impl(q, k, cos, sin, q_out, k_out, q_rows, k_rows, head_dim, seq_len, sb, sh, ss, sd, gid);
}
