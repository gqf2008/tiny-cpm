// Fused SwiGLU: out = silu(gate) * up, as a single Metal kernel.
//
// Mirrors mlx-audio's `@mx.compile def swiglu(gate, x): return silu(gate) * x`
// (talker.py:317). The composite candle path is 2 kernels (silu, then mul) per
// MLP; at m=1 decode those launches are pure overhead. Fusing to one launch is
// the candle equivalent of mlx's @mx.compile for this op.
//
//   silu(g) = g / (1 + exp(-g))  =  g * sigmoid(g)
//   out[i]  = silu(gate[i]) * up[i]
//
// gate/up/out share the same flat length (b * seq * intermediate). One thread per
// element; the grid is 1-D over the flattened element count.
#include <metal_stdlib>
using namespace metal;

template <typename T>
static void swiglu_fused_impl(
    device const T* gate [[buffer(0)]],
    device const T* up [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant int64_t& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if ((int64_t)gid >= numel) return;
    const float g = (float)gate[gid];
    const float u = (float)up[gid];
    const float silu = g / (1.0f + exp(-g));
    out[gid] = (T)(silu * u);
}

[[host_name("swiglu_fused_f32")]]
kernel void swiglu_fused_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int64_t& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    swiglu_fused_impl(gate, up, out, numel, gid);
}

[[host_name("swiglu_fused_bf16")]]
kernel void swiglu_fused_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant int64_t& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    swiglu_fused_impl(gate, up, out, numel, gid);
}
