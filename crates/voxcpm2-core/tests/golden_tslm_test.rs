//! Golden test: compare Rust Candle TSLM layer 0 forward against PyTorch.
//!
//! Prerequisites:
//!   1. python scripts/golden_tslm.py
//!   2. cargo test --test golden_tslm_test -- --nocapture

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;
use std::path::Path;

/// Correct RMSNorm: normalizes over the LAST dimension.
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let ndim = x.shape().dims().len();
    let last_dim = ndim - 1;
    let norm_variance = x_f32.sqr()?.mean_keepdim(last_dim)?;
    let inv_norm = (norm_variance + eps)?.sqrt()?.recip()?;
    let y = x_f32.broadcast_mul(&inv_norm)?;
    y.broadcast_mul(weight)
}

/// Apply RoPE with external cos/sin tables (from golden reference).
fn apply_rope_cos_sin(
    q: &Tensor, k: &Tensor,
    cos: &Tensor, sin: &Tensor,  // [seq_len, half]
) -> candle_core::Result<(Tensor, Tensor)> {
    // q,k: [B, S, H, D]; cos,sin: [S, D/2]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;  // [1, S, 1, D/2]
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;
    let half = q.dim(3)? / 2;
    let q1 = q.narrow(3, 0, half)?;
    let q2 = q.narrow(3, half, half)?;
    let k1 = k.narrow(3, 0, half)?;
    let k2 = k.narrow(3, half, half)?;
    let q_rot = Tensor::cat(
        &[&(q1.broadcast_mul(&cos)? - q2.broadcast_mul(&sin)?)?,
          &(q1.broadcast_mul(&sin)? + q2.broadcast_mul(&cos)?)?], 3)?;
    let k_rot = Tensor::cat(
        &[&(k1.broadcast_mul(&cos)? - k2.broadcast_mul(&sin)?)?,
          &(k1.broadcast_mul(&sin)? + k2.broadcast_mul(&cos)?)?], 3)?;
    Ok((q_rot, k_rot))
}

/// Cosine similarity.
fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() { return 0.0; }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    if na < 1e-30 || nb < 1e-30 { return 0.0; }
    dot / (na * nb)
}

fn tensor_cosine_sim(a: &Tensor, b: &Tensor) -> f64 {
    cosine_sim(
        &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    )
}

fn load_golden(golden: &HashMap<String, Tensor>, key: &str, expected: &[usize]) -> Tensor {
    let t = golden.get(key).unwrap_or_else(|| panic!("missing golden '{key}'"));
    let actual: Vec<usize> = t.shape().dims().to_vec();
    assert_eq!(&actual, expected, "golden '{key}': expected {expected:?}, got {actual:?}");
    t.clone()
}

fn lin_from_golden(golden: &HashMap<String, Tensor>, key: &str, d_in: usize, d_out: usize) -> Linear {
    let w = load_golden(golden, &format!("{key}.weight"), &[d_out, d_in]);
    // Check if bias exists
    let bias_key = format!("{key}.bias");
    let bias = golden.get(&bias_key).map(|t| {
        let actual: Vec<usize> = t.shape().dims().to_vec();
        assert_eq!(actual, [d_out], "golden '{bias_key}': expected [{d_out}], got {actual:?}");
        t.clone()
    });
    Linear::new(w, bias)
}

// ── TSLM Layer 0 forward using crate components ────────────────────────────

fn tslm_layer0_forward(
    x: &Tensor,  // [1, 3, 2048] embed_out
    golden: &HashMap<String, Tensor>,
    cos: &Tensor,  // [3, 64] golden cos table
    sin: &Tensor,  // [3, 64] golden sin table
    eps: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> candle_core::Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();

    // ── RMSNorm w/ golden weights ──
    let ln_w = load_golden(golden, "base_lm.layers.0.input_layernorm.weight", &[2048]);
    let h = rms_norm(x, &ln_w, eps)?;
    out.insert("after_input_layernorm".into(), h.clone());

    // ── QKV projections ──
    // q: 2048→2048, k: 2048→256, v: 2048→256 (no bias)
    let q_lin = lin_from_golden(golden, "base_lm.layers.0.self_attn.q_proj", 2048, 2048);
    let k_lin = lin_from_golden(golden, "base_lm.layers.0.self_attn.k_proj", 2048, 256);
    let v_lin = lin_from_golden(golden, "base_lm.layers.0.self_attn.v_proj", 2048, 256);
    let q = q_lin.forward(&h)?;  // [1, 3, 2048]
    let k = k_lin.forward(&h)?;  // [1, 3, 256]
    let v = v_lin.forward(&h)?;  // [1, 3, 256]
    out.insert("q_proj_out".into(), q.clone());
    out.insert("k_proj_out".into(), k.clone());
    out.insert("v_proj_out".into(), v.clone());

    // ── RoPE using golden cos/sin tables ──
    // Python applies RoPE after transpose to [B, H, T, D], so we must follow
    // the exact same order to match.
    let (_b, seq_len, _hidden) = x.shape().dims3()?;

    let q_heads = q.reshape((1, seq_len, num_heads, head_dim))?; // [1, 3, 16, 128]
    let k_heads = k.reshape((1, seq_len, num_kv_heads, head_dim))?; // [1, 3, 2, 128]

    // Apply RoPE using golden cos/sin tables (loaded from Python reference).
    // Python applies before transpose, so q,k are in [B, H, T, D] after RoPE.
    // We apply in [B, T, H, D], so transpose to match Python layout.
    let (q_rot, k_rot) = apply_rope_cos_sin(&q_heads, &k_heads, cos, sin)?;
    out.insert("q_rope_out".into(), q_rot.transpose(1, 2)?);  // [1, 16, 3, 128]
    out.insert("k_rope_out".into(), k_rot.transpose(1, 2)?);  // [1, 2, 3, 128]

    // ── GQA attention ──
    // Transpose to [B, H, T, D] for SDPA (Python does apply_rope before transpose,
    // but the reshape is [B,T,H,D]→[B,H,T,D] in Python. In our Rust we use
    // [B,T,H,D] layout throughout, so we transpose here for SDPA.)
    let q_rot_t = q_rot.transpose(1, 2)?;  // [1, 16, 3, 128]
    let k_rot_t = k_rot.transpose(1, 2)?;  // [1, 2, 3, 128]

    let group_size = num_heads / num_kv_heads;
    // Expand KV heads: [1, 2, 3, 128] → [1, 16, 3, 128]
    let k_exp = k_rot_t.unsqueeze(2)?
        .expand((1, num_kv_heads, group_size, seq_len, head_dim))?
        .reshape((1, num_heads, seq_len, head_dim))?;
    let v_exp = v.reshape((1, seq_len, num_kv_heads, head_dim))?.transpose(1, 2)?
        .unsqueeze(2)?
        .expand((1, num_kv_heads, group_size, seq_len, head_dim))?
        .reshape((1, num_heads, seq_len, head_dim))?;

    // SDPA
    let scale = (head_dim as f64).powf(-0.5);
    let attn = (q_rot_t.matmul(&k_exp.transpose(2, 3)?)? * scale)?; // [1, 16, 3, 3]
    let attn = candle_nn::ops::softmax(&attn, 3)?;
    let attn_out = attn.matmul(&v_exp)?;  // [1, 16, 3, 128]

    // Transpose back and O projection
    let attn_out = attn_out.transpose(1, 2)?.reshape((1, seq_len, num_heads * head_dim))?; // [1, 3, 2048]
    let o_lin = lin_from_golden(golden, "base_lm.layers.0.self_attn.o_proj", 2048, 2048);
    let attn_out = o_lin.forward(&attn_out)?;
    out.insert("attn_output".into(), attn_out.clone());

    // Residual
    let h = (x + &attn_out)?;
    out.insert("after_attention_residual".into(), h.clone());

    // ── Post-attention RMSNorm ──
    let post_ln_w = load_golden(golden, "base_lm.layers.0.post_attention_layernorm.weight", &[2048]);
    let h_norm = rms_norm(&h, &post_ln_w, eps)?;
    out.insert("after_post_layernorm".into(), h_norm.clone());

    // ── SwiGLU MLP ──
    let gate_lin = lin_from_golden(golden, "base_lm.layers.0.mlp.gate_proj", 2048, 6144);
    let up_lin = lin_from_golden(golden, "base_lm.layers.0.mlp.up_proj", 2048, 6144);
    let down_lin = lin_from_golden(golden, "base_lm.layers.0.mlp.down_proj", 6144, 2048);
    let gate_h = candle_nn::ops::silu(&gate_lin.forward(&h_norm)?)?;
    let up_h = up_lin.forward(&h_norm)?;
    let mlp_out = down_lin.forward(&(gate_h * up_h)?)?;
    out.insert("mlp_output".into(), mlp_out.clone());

    // Final residual
    out.insert("layer_output".into(), (h + mlp_out)?);
    Ok(out)
}

// ── Test ──────────────────────────────────────────────────────────────────

#[test]
fn golden_tslm_layer0() -> anyhow::Result<()> {
    let dev = Device::Cpu;

    let golden_path = Path::new("../../golden/tslm_layer0.safetensors");
    assert!(golden_path.exists(),
        "golden file not found; run: python scripts/golden_tslm.py");
    let golden = candle_core::safetensors::load(golden_path, &dev)?;

    // Config
    let eps = load_golden(&golden, "config_rms_norm_eps", &[1]).to_vec1::<f64>()?[0];
    let num_heads = load_golden(&golden, "config_num_heads", &[1]).to_vec1::<i32>()?[0] as usize;
    let num_kv_heads = load_golden(&golden, "config_num_kv_heads", &[1]).to_vec1::<i32>()?[0] as usize;
    let head_dim = load_golden(&golden, "config_head_dim", &[1]).to_vec1::<i32>()?[0] as usize;

    println!("Config: heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} eps={eps}");

    // Input
    let embed_out = load_golden(&golden, "embed_output", &[1, 3, 2048]);

    // RoPE tables (from Python reference)
    let cos = load_golden(&golden, "rope_cos", &[3, 64]);
    let sin = load_golden(&golden, "rope_sin", &[3, 64]);

    // Run Candle forward
    let rust_out = tslm_layer0_forward(
        &embed_out, &golden, &cos, &sin, eps, num_heads, num_kv_heads, head_dim,
    )?;

    // Compare
    let checks: &[(&str, &str, &[usize])] = &[
        ("after_input_layernorm", "golden_after_input_layernorm", &[1, 3, 2048]),
        ("attn_output", "golden_attn_output", &[1, 3, 2048]),
        ("after_attention_residual", "golden_after_attention_residual", &[1, 3, 2048]),
        ("after_post_layernorm", "golden_after_post_layernorm", &[1, 3, 2048]),
        ("mlp_output", "golden_mlp_output", &[1, 3, 2048]),
        ("layer_output", "golden_layer_output", &[1, 3, 2048]),
    ];

    let mut min_sim = 1.0f64;
    for (rk, gk, shape) in checks {
        let r = rust_out.get(*rk).unwrap();
        let g = load_golden(&golden, gk, shape);
        let sim = tensor_cosine_sim(r, &g);
        println!("  {rk}: {sim:.8}");
        min_sim = min_sim.min(sim);
    }

    // Also compare RoPE and QKV projection outputs
    let rope_checks: &[(&str, &str, &[usize])] = &[
        ("q_rope_out", "golden_q_rope_out", &[1, 16, 3, 128]),
        ("k_rope_out", "golden_k_rope_out", &[1, 2, 3, 128]),
        ("q_proj_out", "golden_q_proj_out", &[1, 3, 2048]),
        ("k_proj_out", "golden_k_proj_out", &[1, 3, 256]),
        ("v_proj_out", "golden_v_proj_out", &[1, 3, 256]),
    ];
    for (rk, gk, shape) in rope_checks {
        let r = rust_out.get(*rk).unwrap();
        let g = load_golden(&golden, gk, shape);
        let sim = tensor_cosine_sim(r, &g);
        println!("  {rk}: {sim:.8}");
        min_sim = min_sim.min(sim);
    }

    println!("\nMin cos_sim = {min_sim:.8}");
    assert!(min_sim > 0.99, "cosine similarity {min_sim:.8} < 0.99");
    println!("✓ Golden TSLM layer 0 test PASSED");
    Ok(())
}
