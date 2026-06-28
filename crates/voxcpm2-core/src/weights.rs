//! Model weight loading from safetensors.
//!
//! Handles loading `model.safetensors` (BF16, main model),
//! `audiovae.safetensors` (F32, with weight-norm fusion),
//! and creating VarBuilders for each model component.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

/// Load `model.safetensors` and return a VarBuilder covering all main model tensors.
///
/// CPU backend does NOT support BF16 matmul, so tensors are converted to F32.
/// On CUDA (with `--features cuda`), BF16 matmul works natively and calling
/// `.to_dtype(DType::BF16)` would match the model's storage format better.
pub fn load_main_vb<'a>(model_dir: &'a Path, device: &'a Device) -> Result<VarBuilder<'a>> {
    let path = model_dir.join("model.safetensors");
    if !path.exists() {
        return Err(candle_core::Error::Msg(format!(
            "model.safetensors not found at {}",
            path.display()
        )));
    }
    let mut tensors = candle_core::safetensors::load(path, device)?;
    // On CPU we must convert BF16→F32 (CPU matmul lacks BF16 support).
    // On CUDA we keep BF16 for memory efficiency.
    let use_bf16 = matches!(device, Device::Cuda(_));
    if !use_bf16 {
        for (_, t) in tensors.iter_mut() {
            if t.dtype() == DType::BF16 {
                *t = t.to_dtype(DType::F32)?;
            }
        }
    }
    let default_dtype = if use_bf16 { DType::BF16 } else { DType::F32 };
    Ok(VarBuilder::from_tensors(tensors, default_dtype, device))
}

/// Load `audiovae.safetensors` with weight-norm fusion and return a VarBuilder.
///
/// The original `.pth` file uses PyTorch's `weight_norm` which splits weights
/// into `weight_g` (per-channel scalar) and `weight_v` (actual kernel).
/// This function fuses them into a single `weight` tensor compatible with
/// Candle's `Conv1d`.
pub fn load_audiovae_vb<'a>(model_dir: &'a Path, device: &'a Device) -> Result<VarBuilder<'a>> {
    let path = model_dir.join("audiovae.safetensors");
    if !path.exists() {
        return Err(candle_core::Error::Msg(format!(
            "audiovae.safetensors not found at {}",
            path.display()
        )));
    }
    let mut tensors = candle_core::safetensors::load(path, device)?;
    fuse_weight_norm(&mut tensors)?;
    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

/// Fuse weight_g/weight_v pairs into single weight tensors.
///
/// PyTorch weight_norm stores:
/// - `layer.weight_g`: shape [out_channels, 1, 1] (per-channel scalar)
/// - `layer.weight_v`: shape [out_channels, in_channels, kernel_size]
///
/// Actual weight = weight_g * weight_v / ||weight_v||_2
/// where ||·||_2 is computed over the non-channel dimensions (in_channels × kernel_size).
///
/// After fusion, the entry `layer.weight` is added and the pair is removed.
fn fuse_weight_norm(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    let keys: Vec<String> = tensors.keys().cloned().collect();
    let mut fused = 0usize;

    for key in &keys {
        if let Some(base) = key.strip_suffix(".weight_g") {
            let v_key = format!("{}.weight_v", base);
            if let Some(weight_v) = tensors.remove(&v_key) {
                if let Some(weight_g) = tensors.remove(key) {
                    // weight_g: [out, 1, 1] or [out], weight_v: [out, in, k]
                    // Compute ||v||_2 over [in, k] dims, keepdim → [out, 1, 1]
                    let v_sq = weight_v.sqr()?;
                    // Chain sum_keepdim over dims 1 and 2 (non-out-channel dims)
                    let v_norm = v_sq.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
                    // Normalize: w_correct = g * (v / ||v||)
                    let weight = weight_g.broadcast_mul(&weight_v.broadcast_div(&v_norm)?)?;
                    tensors.insert(format!("{}.weight", base), weight);
                    fused += 1;
                }
            }
        }
    }

    if fused == 0 {
        eprintln!("[fuse_weight_norm] WARNING: no weight_g/weight_v pairs found — already fused?");
    } else {
        eprintln!("[fuse_weight_norm] fused {fused} weight_norm pairs (with v/||v|| normalization)");
    }

    Ok(())
}

/// Load lm_to_dit_proj and res_to_dit_proj as standalone Linear layers.
/// These project TSLM (2048) and RALM (2048) outputs to DiT hidden (1024).
pub fn load_text_to_dit_projections(
    vb: &VarBuilder<'_>,
) -> Result<(candle_nn::Linear, candle_nn::Linear)> {
    let lm_to_dit = candle_nn::linear(2048, 1024, vb.pp("lm_to_dit_proj"))?;
    let res_to_dit = candle_nn::linear(2048, 1024, vb.pp("res_to_dit_proj"))?;
    Ok((lm_to_dit, res_to_dit))
}

/// Load audiovae decoder tensors as a raw HashMap for manual access.
///
/// This bypasses VarBuilder to handle 3D alpha tensors ([1, C, 1])
/// and ConvTranspose1d weights that VarBuilder cannot represent.
pub fn load_audiovae_decoder_tensors(
    model_dir: &Path,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let path = model_dir.join("audiovae.safetensors");
    let mut tensors = candle_core::safetensors::load(path, device)?;
    fuse_weight_norm(&mut tensors)?;
    // Keep only decoder.* tensors
    tensors.retain(|k, _| k.starts_with("decoder."));
    Ok(tensors)
}

/// Quick sanity: load model.safetensors and report tensor count.
pub fn inspect_main_weights(model_dir: &Path, device: &Device) -> Result<usize> {
    let tensors = candle_core::safetensors::load(model_dir.join("model.safetensors"), device)?;
    Ok(tensors.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_weight_norm_fusion() {
        let dev = Device::Cpu;
        // Simulate weight_norm pairs: weight_g [2,1,1], weight_v [2,3,5]
        let weight_g = Tensor::from_slice(&[2.0f32, 3.0], &[2, 1, 1], &dev).unwrap();
        let weight_v = Tensor::from_slice(
            &(0..30).map(|i| i as f32).collect::<Vec<_>>(),
            &[2, 3, 5],
            &dev,
        )
        .unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("conv.weight_g".into(), weight_g);
        tensors.insert("conv.weight_v".into(), weight_v);
        tensors.insert(
            "conv.bias".into(),
            Tensor::zeros(&[2], DType::F32, &dev).unwrap(),
        );

        fuse_weight_norm(&mut tensors).unwrap();

        // Should have conv.weight, not conv.weight_g or conv.weight_v
        assert!(tensors.contains_key("conv.weight"));
        assert!(!tensors.contains_key("conv.weight_g"));
        assert!(!tensors.contains_key("conv.weight_v"));
        assert_eq!(tensors["conv.weight"].shape().dims(), &[2, 3, 5]);
    }

    #[test]
    #[ignore = "requires model.safetensors in models/VoxCPM2/ (4.6 GB RAM)"]
    fn test_load_main_weights() -> Result<()> {
        let dev = Device::Cpu;
        // Try workspace-root relative path first, then crate-root
        let model_dir = if Path::new("models/VoxCPM2/model.safetensors").exists() {
            Path::new("models/VoxCPM2")
        } else if Path::new("../../models/VoxCPM2/model.safetensors").exists() {
            Path::new("../../models/VoxCPM2")
        } else {
            return Err(candle_core::Error::Msg(
                "model.safetensors not found -- run from workspace root".into(),
            ));
        };
        let vb = load_main_vb(model_dir, &dev)?;
        let count = inspect_main_weights(model_dir, &dev)?;
        assert_eq!(count, 577, "model.safetensors has 577 tensors");

        // Verify we can access key components by prefix
        let _base_lm = vb.pp("base_lm");
        let _residual_lm = vb.pp("residual_lm");
        let _feat_decoder = vb.pp("feat_decoder");
        let _lm_to_dit = vb.pp("lm_to_dit_proj");
        Ok(())
    }

    #[test]
    #[ignore = "requires audiovae.safetensors in models/VoxCPM2/"]
    fn test_load_audiovae_weights() -> Result<()> {
        let dev = Device::Cpu;
        let model_dir = if Path::new("models/VoxCPM2").exists() {
            Path::new("models/VoxCPM2")
        } else {
            Path::new("../../models/VoxCPM2")
        };
        let vb = load_audiovae_vb(model_dir, &dev)?;
        let _decoder = vb.pp("decoder");
        Ok(())
    }
}
