//! Standalone Q4_K GEMV kernel spike: quantize a (K, N) weight to 4-bit blocks
//! (32 elems: f32 min + f32 scale + 4 u32 packed) and run a custom cubecl
//! GEMV on the wgpu/Metal runtime. Verifies numerics vs CPU f32 matmul and
//! measures per-GEMV latency.

use cubecl::prelude::*;
use cubecl::wgpu::{MslCompiler, WgpuRuntime};
type RT = WgpuRuntime<MslCompiler>;

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

/// Quantize a row-major (n, k) f32 weight: per 32-elem block, min + scale=(max-min)/15
/// + 16 bytes of 4-bit values (packed 8/u32, low nibble first).
fn quantize_q4k(w: &[f32], k: usize, n: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
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

fn main() {
    let k = 2048usize; // talker hidden
    let n = 2560usize; // fused qkv rows: 2048 + 256 + 256
    let blocks = k / 32;

    // deterministic pseudo-random weights (talker-like magnitude ~0.02)
    let mut seed = 42u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let w: Vec<f32> = (0..k * n).map(|_| rnd() * 0.02).collect();
    let x: Vec<f32> = (0..k).map(|_| rnd() * 0.5).collect();

    // CPU f64 reference
    let y_ref: Vec<f64> = (0..n)
        .map(|row| {
            (0..k)
                .map(|i| w[row * k + i] as f64 * x[i] as f64)
                .sum::<f64>()
        })
        .collect();

    let (values, scales, mins) = quantize_q4k(&w, k, n);
    let deq_err: f64 = (0..n)
        .map(|row| {
            let mut e = 0.0f64;
            for i in 0..k {
                let b = i / 32;
                let q = (values[row * blocks * 4 + b * 4 + (i % 32) / 8] >> ((i % 8) * 4)) & 0xF;
                let v = mins[row * blocks + b] + q as f32 * scales[row * blocks + b];
                e = e.max((v as f64 - w[row * k + i] as f64).abs());
            }
            e
        })
        .fold(0.0f64, f64::max);
    println!("quantize max|Δ| per element: {deq_err:.5}");

    let client = RT::client(&Default::default());
    let x_h = client.create_from_slice(f32::as_bytes(&x));
    let v_h = client.create_from_slice(u32::as_bytes(&values));
    let s_h = client.create_from_slice(f32::as_bytes(&scales));
    let m_h = client.create_from_slice(f32::as_bytes(&mins));
    let o_h = client.empty(n * 4);

    unsafe {
        q4k_gemv::launch_unchecked::<f32, RT>(
            &client,
            CubeCount::Static(((n + 255) / 256) as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x_h.clone(), k),
            BufferArg::from_raw_parts(v_h.clone(), values.len()),
            BufferArg::from_raw_parts(s_h.clone(), scales.len()),
            BufferArg::from_raw_parts(m_h.clone(), mins.len()),
            BufferArg::from_raw_parts(o_h.clone(), n),
            blocks as u32,
        )
    };

    let bytes = client.read_one(o_h.clone()).unwrap();
    let y_gpu: Vec<f32> = f32::from_bytes(&bytes).to_vec();
    println!("y_gpu[0..8]  {:?}", &y_gpu[..8]);
    println!(
        "y_ref[0..8]  {:?}",
        &y_ref[..8].iter().map(|&v| v as f32).collect::<Vec<_>>()
    );
    let rel: Vec<f64> = (0..n)
        .map(|i| (y_gpu[i] as f64 - y_ref[i]).abs() / y_ref[i].abs().max(1e-9))
        .collect();
    let max_rel = rel.iter().copied().fold(0.0f64, f64::max);
    let mean_rel = rel.iter().sum::<f64>() / rel.len() as f64;
    println!("GEMV vs CPU: max rel err {max_rel:.4e}, mean {mean_rel:.4e}");

    // Latency: 200 launches, no readback in between.
    let t0 = std::time::Instant::now();
    for _ in 0..200 {
        unsafe {
            q4k_gemv::launch_unchecked::<f32, RT>(
                &client,
                CubeCount::Static(((n + 255) / 256) as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(x_h.clone(), k),
                BufferArg::from_raw_parts(v_h.clone(), values.len()),
                BufferArg::from_raw_parts(s_h.clone(), scales.len()),
                BufferArg::from_raw_parts(m_h.clone(), mins.len()),
                BufferArg::from_raw_parts(o_h.clone(), n),
                blocks as u32,
            )
        };
    }
    let t = t0.elapsed();
    println!(
        "Q4K GEMV ({k}x{n}): {:.2} us/launch (200 launches in {t:?})",
        t.as_secs_f64() * 1e6 / 200.0
    );
}
