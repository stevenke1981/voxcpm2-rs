//! Integration tests that load real model weights from `models/VoxCPM2/`.
//!
//! These tests are `#[ignore]` by default and require:
//! - model.safetensors (577 tensors, BF16, ~4.58GB RAM for full load)
//! - audiovae.safetensors (312 tensors, F32, with weight_norm)

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

/// Helper: initialize device (Cpu unless CUDA/Metal available).
fn get_device() -> Device {
    if let Ok(d) = Device::cuda_if_available(0) {
        d
    } else if let Ok(d) = Device::metal_if_available(0) {
        d
    } else {
        Device::Cpu
    }
}

const MODEL_DIR: &str = "models/VoxCPM2";

#[test]
#[ignore = "requires model.safetensors (~4.6 GB RAM for full load)"]
fn test_main_weights_load() {
    let dev = get_device();
    let vb = voxcpm2_core::weights::load_main_vb(MODEL_DIR.as_ref(), &dev).unwrap();

    // Verify key weight exists with correct shape
    let embed = vb.pp("base_lm.embed_tokens");
    let _w: Tensor = embed.get(&[73448, 2048], "weight").unwrap();
}

#[test]
#[ignore = "requires model.safetensors and audiovae.safetensors"]
fn test_audiovae_weights_load() {
    let dev = get_device();
    let vb = voxcpm2_core::weights::load_audiovae_vb(MODEL_DIR.as_ref(), &dev).unwrap();

    // After weight-norm fusion, should have decoder.model.0.weight
    let decoder = vb.pp("decoder");
    let _w: Tensor = decoder.get(&[64, 1, 7], "model.0.weight").unwrap();
}

#[test]
#[ignore = "requires model.safetensors, slow (full TSLM forward)"]
fn test_tslm_load_with_real_weights() -> anyhow::Result<()> {
    let dev = get_device();
    let vb = voxcpm2_core::weights::load_main_vb(MODEL_DIR.as_ref(), &dev)?;

    // Build config from the real config.json
    let config = voxcpm2_core::config::VoxConfig::load(MODEL_DIR.as_ref())?;

    // Load TSLM with real weights
    let mut tslm = voxcpm2_core::models::TSLM::load(
        &vb.pp("base_lm"),
        &config.lm_config,
        &dev,
        false,
    )?;

    // Tokenize a short prompt
    let tokenizer = voxcpm2_core::tokenizer::VoxTokenizer::load(MODEL_DIR.as_ref())?;
    let tokens = tokenizer.encode("Hello world")?;
    let input_ids = Tensor::from_slice(&tokens, &[1, tokens.len()], &dev)?
        .to_dtype(DType::I64)?;

    // Forward pass
    let h = tslm.forward(&input_ids, 0)?;

    // Verify output shape
    assert_eq!(h.dims(), &[1, tokens.len(), config.lm_config.hidden_size]);
    println!("TSLM forward OK: shape {:?}", h.dims());

    Ok(())
}

#[test]
#[ignore = "requires all model weights, full pipeline integration"]
fn test_full_pipeline_smoke() -> anyhow::Result<()> {
    use voxcpm2_core::models::*;

    let dev = get_device();
    let config = voxcpm2_core::config::VoxConfig::load(MODEL_DIR.as_ref())?;
    let vb = voxcpm2_core::weights::load_main_vb(MODEL_DIR.as_ref(), &dev)?;
    let tokenizer = voxcpm2_core::tokenizer::VoxTokenizer::load(MODEL_DIR.as_ref())?;

    // --- 1. Tokenize ---
    let tokens = tokenizer.encode("Hello world")?;
    let input_ids = Tensor::from_slice(&tokens, &[1, tokens.len()], &dev)?
        .to_dtype(DType::I64)?;

    // --- 2. TSLM forward ---
    let mut tslm = TSLM::load(&vb.pp("base_lm"), &config.lm_config, &dev, false)?;
    let h_tslm = tslm.forward(&input_ids, 0)?;
    assert_eq!(h_tslm.dims(), &[1, tokens.len(), 2048]);
    println!("TSLM: {:?}", h_tslm.dims());

    // --- 3. RALM forward ---
    let mut ralm = RALM::load(&vb.pp("residual_lm"), &config.lm_config, &dev, false)?;
    let h_ralm = ralm.forward(&h_tslm, 0)?;
    assert_eq!(h_ralm.dims(), &[1, tokens.len(), 2048]);
    println!("RALM: {:?}", h_ralm.dims());

    // --- 4. Project to Dit hidden dim ---
    let (lm_to_dit, res_to_dit) =
        voxcpm2_core::weights::load_text_to_dit_projections(&vb)?;
    let cond_1024 = (lm_to_dit.forward(&h_tslm)? + res_to_dit.forward(&h_ralm)?)?;
    assert_eq!(cond_1024.dims(), &[1, tokens.len(), 1024]);
    println!("Text→DiT cond: {:?}", cond_1024.dims());

    // --- 5. LocDiT ---
    // For now, use a dummy cond with feat_dim (64) since cond_proj expects 64-dim.
    // In the real pipeline, the 1024-dim text cond needs further processing to 64-dim.
    let feat_dim = config.feat_dim;
    let hidden_dim = config.dit_config.hidden_dim;
    let seq_len = tokens.len();
    let cond_dit = Tensor::zeros(&[1, seq_len, feat_dim], DType::F32, &dev)?;

    let mut dit = LocDiT::load(&vb.pp("feat_decoder"), &config.dit_config, feat_dim)?;

    // Test the estimator forward with a single patch
    let patch_size = 4;
    let x = Tensor::randn(0.0f32, 1.0, &[1, feat_dim, patch_size], &dev)?;
    // mu = dit_hidden (2 * hidden_dim)
    let mu = Tensor::randn(0.0f32, 1.0, &[1, 2 * hidden_dim], &dev)?;
    let t = Tensor::full(0.5f32, &[1], &dev)?;
    let cond = Tensor::randn(0.0f32, 1.0, &[1, feat_dim, patch_size], &dev)?;
    let dt = Tensor::full(0.1f32, &[1], &dev)?;
    let out = dit.forward(&x, &mu, &t, &cond, &dt)?;
    assert_eq!(out.dims(), &[1, feat_dim, patch_size], "LocDiT forward shape");

    println!("Pipeline structure OK (LocDiT forward shape verified)");
    Ok(())
}
