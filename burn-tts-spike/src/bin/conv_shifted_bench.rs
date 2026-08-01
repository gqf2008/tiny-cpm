//! Micro-bench: burn conv1d (im2col) vs shifted-window matmul equiv on Metal.
use burn::tensor::{Tensor, TensorData, Shape, ops::ConvOptions, module};

fn causal_conv_shifted(
    x: &Tensor<3, burn::tensor::Float>,
    w: &Tensor<3, burn::tensor::Float>, // (out, in, k)
    b: Option<Tensor<1, burn::tensor::Float>>,
    dilation: usize,
) -> Tensor<3, burn::tensor::Float> {
    let [b0, cin, t] = x.dims();
    let k = w.dims()[2];
    let out_c = w.dims()[0];
    let device = x.device();
    let mut acc: Option<Tensor<3, burn::tensor::Float>> = None;
    for q in 0..k {
        let shift = q * dilation;
        let xq = if shift == 0 {
            x.clone()
        } else {
            let zeros = Tensor::<3, burn::tensor::Float>::zeros([b0, cin, shift], &device);
            Tensor::cat(vec![zeros, x.clone().narrow(2, 0, t - shift)], 2)
        };
        let wq = w.clone().narrow(2, k - 1 - q, 1).reshape([out_c, cin]).swap_dims(0, 1); // (in, out), kernel reversed
        let m = xq.swap_dims(1, 2).reshape([b0 * t, cin]).matmul(wq).reshape([b0, t, out_c]); // (B, T, out)
        acc = Some(match acc {
            None => m,
            Some(a) => a + m,
        });
    }
    let mut y = acc.unwrap().swap_dims(1, 2); // (B, out, T)
    if let Some(bias) = b {
        y = y + bias.clone().reshape([1, out_c, 1]);
    }
    y
}

fn main() {
    let device = burn::tensor::Device::default();
    let cases: Vec<(usize, usize, usize, usize, usize)> = vec![
        // (cin, cout, t, k, dilation) — residual unit shapes at grown T
        (192, 192, 12800, 7, 9),
        (96, 96, 38400, 7, 3),
        (192, 192, 3200, 7, 9),
    ];
    let mut seed = 7u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    for (cin, cout, t, k, dil) in cases {
        let x_v: Vec<f32> = (0..cin * t).map(|_| rnd() * 0.05).collect();
        let w_v: Vec<f32> = (0..cout * cin * k).map(|_| rnd() * 0.05).collect();
        let b_v: Vec<f32> = (0..cout).map(|_| rnd() * 0.01).collect();
        let x: Tensor<3, burn::tensor::Float> = Tensor::from_data(TensorData::new(x_v, Shape::from([1, cin, t])), &device);
        let w: Tensor<3, burn::tensor::Float> = Tensor::from_data(TensorData::new(w_v, Shape::from([cout, cin, k])), &device);
        let b: Tensor<1, burn::tensor::Float> = Tensor::from_data(TensorData::new(b_v, Shape::from([cout])), &device);

        // causal left pad (k-1)*dil
        let pad = (k - 1) * dil;
        let xp = Tensor::cat(vec![Tensor::<3, burn::tensor::Float>::zeros([1, cin, pad], &device), x.clone()], 2);
        let opts = ConvOptions::new([1], [0], [dil], 1);
        let y1 = module::conv1d(xp.clone(), w.clone(), Some(b.clone()), opts.clone());
        let y2 = causal_conv_shifted(&x, &w, Some(b.clone()), dil);
        let d1: Vec<f32> = y1.clone().into_data().to_vec().unwrap();
        let d2: Vec<f32> = y2.clone().into_data().to_vec().unwrap();
        let maxd = d1.iter().zip(d2.iter()).map(|(a, bb)| (a - bb).abs()).fold(0.0f32, f32::max);
        println!("case {cin}x{t} k{k} dil{dil}: out {} vs {}, max|d|={maxd:.2e}", y1.dims()[2], y2.dims()[2]);

        let reps = 3usize;
        let t0 = std::time::Instant::now();
        for _ in 0..reps { let _ = module::conv1d(xp.clone(), w.clone(), Some(b.clone()), opts.clone()); }
        device.sync().unwrap();
        let t1 = std::time::Instant::now();
        let t0b = std::time::Instant::now();
        for _ in 0..reps { let _ = causal_conv_shifted(&x, &w, Some(b.clone()), dil); }
        device.sync().unwrap();
        let t1b = std::time::Instant::now();
        println!("  conv1d(im2col): {:.2} ms/call, shifted-matmul: {:.2} ms/call",
            (t1 - t0).as_secs_f64() * 1e3 / reps as f64,
            (t1b - t0b).as_secs_f64() * 1e3 / reps as f64);
    }
}
