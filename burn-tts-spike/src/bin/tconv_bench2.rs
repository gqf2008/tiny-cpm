//! Verify + bench: transposed conv via zero-insert + shifted causal conv.
use burn::tensor::{Tensor, TensorData, Shape, ops::{ConvOptions, ConvTransposeOptions}, module};

fn pad_zeros(x: Tensor<3, burn::tensor::Float>, left: usize, right: usize) -> Tensor<3, burn::tensor::Float> {
    let [b, c, _t] = x.dims();
    let device = x.device();
    let mut parts = Vec::new();
    if left > 0 { parts.push(Tensor::<3, burn::tensor::Float>::zeros([b, c, left], &device)); }
    parts.push(x);
    if right > 0 { parts.push(Tensor::<3, burn::tensor::Float>::zeros([b, c, right], &device)); }
    Tensor::cat(parts, 2)
}

/// conv1d (out,in,k) stride 1 causal (left pad k-1) via shifted taps.
fn causal_conv_shifted(
    x: Tensor<3, burn::tensor::Float>,
    w: Tensor<3, burn::tensor::Float>,
    b: Option<Tensor<1, burn::tensor::Float>>,
    dilation: usize,
) -> Tensor<3, burn::tensor::Float> {
    let [b0, cin, t] = x.dims();
    let [out_c, _, k] = w.dims();
    let device = x.device();
    let mut acc: Option<Tensor<3>> = None;
    for q in 0..k {
        let shift = q * dilation;
        let xq = if shift == 0 {
            x.clone()
        } else {
            let zeros = Tensor::<3, burn::tensor::Float>::zeros([b0, cin, shift], &device);
            Tensor::cat(vec![zeros, x.clone().narrow(2, 0, t - shift)], 2)
        };
        let wq = w.clone().narrow(2, q, 1).reshape([out_c, cin]).swap_dims(0, 1);
        let m = xq.swap_dims(1, 2).reshape([b0 * t, cin]).matmul(wq).reshape([b0, t, out_c]);
        let m = m.swap_dims(1, 2);
        acc = Some(match acc { None => m, Some(a) => a + m });
    }
    let mut y = acc.unwrap();
    if let Some(bias) = b { y = y + bias.clone().reshape([1, out_c, 1]); }
    y
}

/// zero-insert + causal conv1d (reversed kernel), right-pad k-1.
fn tconv_equiv(
    x: &Tensor<3, burn::tensor::Float>,
    w: &Tensor<3, burn::tensor::Float>, // (in, out, k) conv_transpose layout
    b: Option<Tensor<1, burn::tensor::Float>>,
    stride: usize,
) -> Tensor<3, burn::tensor::Float> {
    let [b0, c, t] = x.dims();
    let k = w.dims()[2];
    let x4 = x.clone().unsqueeze_dim::<4>(3);
    let zeros = Tensor::<4, burn::tensor::Float>::zeros([b0, c, t, stride - 1], &x4.device());
    let inter = Tensor::cat(vec![x4, zeros], 3);
    let x_up = inter.reshape([b0, c, t * stride]).narrow(2, 0, (t - 1) * stride + 1);
    let x_up = pad_zeros(x_up, 0, k - 1);
    causal_conv_shifted(x_up, w.clone().swap_dims(0, 1), b, 1)
}

fn main() {
    let device = burn::tensor::Device::default();
    let cases: Vec<(usize, usize, usize, usize, usize)> = vec![
        (768, 384, 160, 10, 5),
        (384, 192, 800, 8, 4),
        (192, 96, 3200, 6, 3),
    ];
    let mut seed = 7u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    for (cin, cout, t, k, s) in cases {
        let x_v: Vec<f32> = (0..cin * t).map(|_| rnd() * 0.05).collect();
        let w_v: Vec<f32> = (0..cin * cout * k).map(|_| rnd() * 0.05).collect();
        let b_v: Vec<f32> = (0..cout).map(|_| rnd() * 0.01).collect();
        let x: Tensor<3, burn::tensor::Float> = Tensor::from_data(TensorData::new(x_v, Shape::from([1, cin, t])), &device);
        let w: Tensor<3, burn::tensor::Float> = Tensor::from_data(TensorData::new(w_v, Shape::from([cin, cout, k])), &device);
        let b: Tensor<1, burn::tensor::Float> = Tensor::from_data(TensorData::new(b_v, Shape::from([cout])), &device);

        let opts = ConvTransposeOptions::new([s], [0], [0], [1], 1);
        let y1 = module::conv_transpose1d(x.clone(), w.clone(), Some(b.clone()), opts.clone());
        let y2 = tconv_equiv(&x, &w, Some(b.clone()), s);
        let d1: Vec<f32> = y1.clone().into_data().to_vec().unwrap();
        let d2: Vec<f32> = y2.clone().into_data().to_vec().unwrap();
        let md = if d1.len() == d2.len() {
            d1.iter().zip(d2.iter()).map(|(a, bb)| (a - bb).abs()).fold(0.0f32, f32::max)
        } else { -1.0 };
        println!("case {cin}x{t} k{k} s{s}: convT out {} vs equiv {}, max|d|={md:.2e}", y1.dims()[2], y2.dims()[2]);

        let reps = 5usize;
        let t0 = std::time::Instant::now();
        for _ in 0..reps { let _ = module::conv_transpose1d(x.clone(), w.clone(), Some(b.clone()), opts.clone()); }
        device.sync().unwrap();
        let t1 = std::time::Instant::now();
        let t0b = std::time::Instant::now();
        for _ in 0..reps { let _ = tconv_equiv(&x, &w, Some(b.clone()), s); }
        device.sync().unwrap();
        let t1b = std::time::Instant::now();
        println!("  conv_transpose: {:.2} ms, equiv: {:.2} ms",
            (t1 - t0).as_secs_f64() * 1e3 / reps as f64,
            (t1b - t0b).as_secs_f64() * 1e3 / reps as f64);
    }
}
