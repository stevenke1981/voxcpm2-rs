//! FSQ Layer — Finite Scalar Quantization for acoustic feature discretization.
//!
//! Tensor structure (from `model.safetensors` `fsq_layer.*`):
//!
//! fsq_layer.in_proj.weight:  [512, 2048]   Linear(2048, 512, bias=false)
//! fsq_layer.out_proj.weight: [2048, 512]    Linear(512, 2048, bias=false)
//!
//! Quantization:
//!   - scalar_quantization_latent_dim = 512
//!   - scalar_quantization_scale = 9
//!   - Quantize: round(x * scale) / scale  (clamped to [-scale, scale])

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

pub const FSQ_LATENT_DIM: usize = 512;
pub const FSQ_SCALE: f32 = 9.0;

/// Finite Scalar Quantization layer.
///
/// Projects `[B, T, hidden_dim=2048]` → quantized latent space → `[B, T, 2048]`.
pub struct FsqLayer {
    in_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    latent_dim: usize,
    scale: f32,
}

impl FsqLayer {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let in_proj = candle_nn::linear_no_bias(2048, FSQ_LATENT_DIM, vb.pp("fsq_layer.in_proj"))?;
        let out_proj =
            candle_nn::linear_no_bias(FSQ_LATENT_DIM, 2048, vb.pp("fsq_layer.out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            latent_dim: FSQ_LATENT_DIM,
            scale: FSQ_SCALE,
        })
    }

    /// Forward pass with quantization.
    ///
    /// # Args
    /// * `x` — `[B, T, 2048]` (LM hidden states)
    ///
    /// # Returns
    /// * `[B, T, 2048]` (quantized hidden states)
    ///
    /// # Important
    /// Python order: in_proj → tanh → round → out_proj
    /// The tanh MUST come BEFORE rounding to match Python's ScalarQuantizationLayer.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dev = x.device();
        let dtype = x.dtype();
        // Project to latent dim
        let h = self.in_proj.forward(x)?; // [B, T, latent_dim], same dtype as x
                                          // Python: torch.tanh(h) first, THEN round(x * scale) / scale
                                          // tanh clips to [-1, 1] before quantization (matching Python).
        let h = h.tanh()?;
        // Scalar quantization: round to nearest discrete level
        let scale_t = make_tensor_like(self.scale as f32, &[1], dtype, dev)?;
        let h_q = h.broadcast_mul(&scale_t)?; // x * 9
                                              // Round via: floor(x + 0.5) — Python's torch.round behavior
        let half = make_tensor_like(0.5f32, &[self.latent_dim], dtype, dev)?;
        let h_q = h_q.broadcast_add(&half)?.floor()?; // round(x * 9)
        let h_q = h_q.broadcast_div(&scale_t)?; // / 9
                                                // Project back to hidden dim
        self.out_proj.forward(&h_q)
    }
}

/// Create a full tensor in the specified dtype on the specified device.
/// Uses a CPU bridge when the device is CUDA and dtype is non-F32
/// to avoid `to_dtype` CUDA limitations in candle 0.9.2.
fn make_tensor_like<S: Into<candle_core::Shape>>(
    value: f32,
    shape: S,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    let shape = shape.into();
    if !dev.is_cuda() || dtype == DType::F32 {
        Tensor::full(value, &shape, dev)
    } else {
        let cpu_t = Tensor::full(value, &shape, &Device::Cpu)?;
        let cvt = cpu_t.to_dtype(dtype)?;
        cvt.to_device(dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn fsq_shape_test() -> Result<()> {
        let dev = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let fsq = FsqLayer::load(&vb)?;
        let x = Tensor::zeros(&[1, 5, 2048], DType::F32, &dev)?;
        let y = fsq.forward(&x)?;
        assert_eq!(y.dims(), &[1, 5, 2048], "FSQ forward shape");
        Ok(())
    }

    #[test]
    fn fsq_quantization_preserves_zeros() -> Result<()> {
        let dev = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let fsq = FsqLayer::load(&vb)?;
        let x = Tensor::zeros(&[1, 3, 2048], DType::F32, &dev)?;
        let y = fsq.forward(&x)?;
        // With zero weights, in_proj gives zeros, round(0*9)=0, tanh(0)=0, out_proj(0)=0
        let data = y.to_vec3::<f32>()?;
        for batch in &data {
            for row in batch {
                for &val in row {
                    assert!(val.abs() < 1e-5, "zero input should give zero output");
                }
            }
        }
        Ok(())
    }
}
