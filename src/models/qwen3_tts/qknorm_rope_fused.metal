// Fused per-head QK-RMSNorm + RoPE for the Qwen3-TTS decode path.
//
// The QKNormAttention forward does, per q/k: a per-head RMSNorm over head_dim
// (one reduction kernel each), then the RoPE kernel (rope_fused). That's 3
// launches/layer; at m=1 decode, ×5 predictor layers ×15 steps, it's a real
// chunk of the launch-bound floor. This fuses them: ONE kernel does the
// RMSNorm reduction AND the rotary in a single pass over each head's `head_dim`
// elements.
//
// Input layout: the projections reshaped to (b, seq, heads, head_dim) and made
// contiguous — i.e. the layout QK-norm is applied in, BEFORE the transpose(1,2)
// that feeds RoPE/SDPA. Each "row" is one (b, seq, head) triple = head_dim
// contiguous elements. We do RoPE in this same layout: the rotary partner index
// (i ± head_dim/2) stays within the row, and cos/sin are indexed by the row's
// sequence position — layout-independent, so the output is the normed+rotated
// (b, seq, heads, head_dim) tensor the caller then transposes for SDPA.
//
//   RMSNorm:  inv_rms = rsqrt(mean(x^2) + eps);  xn[i] = x[i] * inv_rms * nw[i]
//   RoPE:     out[i]  = xn[i]*cos[i] + rot[i]*sin[i]
//             rot[i]   = i < half ? -xn[i+half] : xn[i-half]   (half = head_dim/2)
//
// One threadgroup per row, head_dim threads (head_dim ≤ 1024). threadgroup
// scratch/bcast are declared in the kernel bodies (MSL forbids threadgroup
// storage in a non-kernel function) and passed down by reference.
#include <metal_stdlib>
using namespace metal;

template <typename T>
static void qknorm_rope_impl(
    device const T* q [[buffer(0)]],
    device const T* k [[buffer(1)]],
    device const float* qn_w [[buffer(2)]],  // (head_dim,) F32 q-norm weight
    device const float* kn_w [[buffer(3)]],  // (head_dim,) F32 k-norm weight
    device const float* cos [[buffer(4)]],   // (seq_len, head_dim) F32
    device const float* sin [[buffer(5)]],   // (seq_len, head_dim) F32
    device T* q_out [[buffer(6)]],
    device T* k_out [[buffer(7)]],
    constant int64_t& q_rows [[buffer(8)]],  // b * seq * q_heads
    constant int64_t& k_rows [[buffer(9)]],  // b * seq * kv_heads
    constant int64_t& head_dim [[buffer(10)]],
    constant int64_t& seq_len [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    threadgroup float* scratch,
    threadgroup float* bcast,
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
    const int64_t row = tgid;
    // Select source/dest/norm-weight by row (q rows first, then k rows).
    device const T* inp;
    device T* outp;
    device const float* nw;
    int64_t r;
    int64_t heads;  // heads-per-batch for the selected tensor
    if (row < q_rows) {
        inp = q; outp = q_out; nw = qn_w; r = row;
        heads = seq_len > 0 ? (q_rows / seq_len) : 1;
    } else if (row < q_rows + k_rows) {
        inp = k; outp = k_out; nw = kn_w; r = row - q_rows;
        heads = seq_len > 0 ? (k_rows / seq_len) : 1;
    } else {
        return;
    }
    const int64_t base = r * head_dim;
    const int half_d = (int)(head_dim / 2);
    // Decompose row within (b, seq, heads) row-major: r = ((b*seq_len)+seq)*heads+head,
    // so seq = (r / heads) % seq_len. (NOT r % seq_len — that's wrong when heads>1.)
    const int64_t seq = (heads > 0 && seq_len > 0) ? ((r / heads) % seq_len) : 0;

    // 1) block-reduce sum(x^2) over head_dim via simdgroup partials.
    float local = 0.0f;
    for (int64_t i = tid; i < head_dim; i += tpg) {
        const float x = (float)inp[base + i];
        local += x * x;
    }
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    local = simd_sum(local);
    if (lane == 0) scratch[sg] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nsg = (tpg + 31) / 32;
    if (tid == 0) {
        float total = 0.0f;
        for (uint sgi = 0; sgi < nsg; sgi++) total += scratch[sgi];
        *bcast = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = rsqrt(*bcast / (float)head_dim + eps);

    // 2) RMSNorm-scale by the norm weight, then RoPE. The rotary partner (i±half)
    // needs the PARTNER's normed value too, so compute both xn[i] and xn[p].
    for (int64_t i = tid; i < head_dim; i += tpg) {
        const int64_t p = (i < (int64_t)half_d) ? (i + half_d) : (i - half_d);
        const float xi = (float)inp[base + i] * inv_rms * nw[i];
        const float xp = (float)inp[base + p] * inv_rms * nw[p];
        const float rot = (i < (int64_t)half_d) ? -xp : xp;
        const float c = cos[seq * head_dim + i];
        const float s = sin[seq * head_dim + i];
        outp[base + i] = (T)(xi * c + rot * s);
    }
}

[[host_name("qknorm_rope_f32")]]
kernel void qknorm_rope_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* qn_w [[buffer(2)]],
    device const float* kn_w [[buffer(3)]],
    device const float* cos [[buffer(4)]],
    device const float* sin [[buffer(5)]],
    device float* q_out [[buffer(6)]],
    device float* k_out [[buffer(7)]],
    constant int64_t& q_rows [[buffer(8)]],
    constant int64_t& k_rows [[buffer(9)]],
    constant int64_t& head_dim [[buffer(10)]],
    constant int64_t& seq_len [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    threadgroup float bcast;
    qknorm_rope_impl(q, k, qn_w, kn_w, cos, sin, q_out, k_out, q_rows, k_rows, head_dim, seq_len, eps, scratch, &bcast, tgid, tid, tpg);
}

[[host_name("qknorm_rope_bf16")]]
kernel void qknorm_rope_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const float* qn_w [[buffer(2)]],
    device const float* kn_w [[buffer(3)]],
    device const float* cos [[buffer(4)]],
    device const float* sin [[buffer(5)]],
    device bfloat* q_out [[buffer(6)]],
    device bfloat* k_out [[buffer(7)]],
    constant int64_t& q_rows [[buffer(8)]],
    constant int64_t& k_rows [[buffer(9)]],
    constant int64_t& head_dim [[buffer(10)]],
    constant int64_t& seq_len [[buffer(11)]],
    constant float& eps [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    threadgroup float bcast;
    qknorm_rope_impl(q, k, qn_w, kn_w, cos, sin, q_out, k_out, q_rows, k_rows, head_dim, seq_len, eps, scratch, &bcast, tgid, tid, tpg);
}
