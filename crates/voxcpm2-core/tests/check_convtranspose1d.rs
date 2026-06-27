//! Verify Candle ConvTranspose1d against Python reference.
//! Run: python scripts/gen_ct_test.py  (generates data/*.npy)
//! Then: cargo test --test check_convtranspose1d --release -- --nocapture

use candle_core::{Device, Module, Tensor};
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR points to the crate root (voxcpm2-core)
    // The workspace root is two levels up
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("data")
}
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig};

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn check_candle_convtranspose1d() {
    let dev = Device::Cpu;
    let test_cases = [
        ("model_5_block_1", 256, 128, 4, 2, 1, 0),
        ("model_6_block_1", 128, 64, 4, 2, 1, 0),
        ("model_7_block_1", 64, 32, 4, 2, 1, 0),
        ("model_2_block_1", 2048, 1024, 16, 8, 4, 0),
        ("model_3_block_1", 1024, 512, 12, 6, 3, 0),
        ("model_4_block_1", 512, 256, 10, 5, 3, 1),
    ];

    for (prefix, in_ch, out_ch, k, s, pad, op) in &test_cases {
        let d = data_dir();
        let weight_path = d.join(format!("{prefix}_w.npy"));
        let bias_path = d.join(format!("{prefix}_b.npy"));
        let input_path = d.join(format!("{prefix}_x.npy"));
        let expected_path = d.join(format!("{prefix}_y.npy"));

        let weight = Tensor::read_npy(weight_path).unwrap().to_device(&dev).unwrap();
        let bias = Tensor::read_npy(bias_path).unwrap().to_device(&dev).unwrap();
        let input = Tensor::read_npy(input_path).unwrap().to_device(&dev).unwrap();
        let expected_t = Tensor::read_npy(expected_path).unwrap().to_device(&dev).unwrap();

        // Check shapes
        assert_eq!(weight.shape().dims(), &[*in_ch, *out_ch, *k],
            "{prefix} weight shape");
        assert_eq!(bias.shape().dims(), &[*out_ch],
            "{prefix} bias shape");
        assert_eq!(input.shape().dims(), &[1, *in_ch, 16],
            "{prefix} input shape");

        let cfg = ConvTranspose1dConfig {
            padding: *pad,
            output_padding: *op,
            stride: *s,
            ..Default::default()
        };
        let conv = ConvTranspose1d::new(weight, Some(bias), cfg);
        let out = conv.forward(&input).unwrap();

        let expected = expected_t.to_vec3::<f32>().unwrap();
        let actual = out.to_vec3::<f32>().unwrap();

        fn flatten_3d(v: &[Vec<Vec<f32>>]) -> Vec<f32> {
            let mut out = Vec::new();
            for b in v { for t in b { for &x in t { out.push(x); } } }
            out
        }
        let expected_flat = flatten_3d(&expected);
        let actual_flat = flatten_3d(&actual);
        let diff = max_diff(&expected_flat, &actual_flat);

        let peak: f32 = out.abs().unwrap().flatten_all().unwrap().max_all().unwrap().to_scalar().unwrap();
        println!("{prefix}: shape={:?}, peak={:.6}, max_diff={:.8} (tolerance=1e-4)",
            out.shape(), peak, diff);

        assert!(
            diff < 1e-4,
            "{prefix} mismatch: max_diff={diff:.8} >= 1e-4"
        );
    }

    println!("\nAll ConvTranspose1d tests PASSED");
}
