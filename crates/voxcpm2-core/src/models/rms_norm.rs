//! RMSNorm — 基礎歸一化層，用於 TSLM / RALM / LocEnc / LocDiT 所有子模組。
//!
//! 對應 `base_lm.layers.*.input_layernorm.weight` 與 `post_attention_layernorm.weight`。

#[cfg(test)]
use candle_core::Device;
use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    /// 從已載入的 weight tensor 建立（shape = [hidden_size]）。
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { weight, eps })
    }

    /// 從 safetensors VarBuilder 載入。
    pub fn load(
        vb: &candle_nn::VarBuilder,
        hidden_size: usize,
        eps: f64,
        prefix: &str,
    ) -> Result<Self> {
        let weight = vb.get(hidden_size, &format!("{prefix}.weight"))?;
        Ok(Self { weight, eps })
    }

    /// RMSNorm: `x * rsqrt(mean(x^2, dim=-1) + eps) * weight`
    ///
    /// # 重要
    /// 必須對最後一維歸一化（特徵維度 D），不能 hardcode dim=1。
    /// - 2D `[T, D]` → dim=1
    /// - 3D `[B, T, D]` → dim=2
    fn last_dim(ndim: usize) -> usize {
        ndim - 1
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f64 = x.to_dtype(DType::F64)?;
        // mean over last dimension (feature dim) — supports both 2D and 3D
        let norm_dim = Self::last_dim(x.shape().dims().len());
        let norm = x_f64.sqr()?.mean_keepdim(norm_dim)?;
        let x_normed = x_f64.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x.dtype())?;
        x_normed.broadcast_mul(&self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_forward_2d() -> Result<()> {
        let dev = Device::Cpu;
        let weight = Tensor::ones(&[4], DType::F32, &dev)?;
        let norm = RMSNorm::new(weight, 1e-5)?;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &dev)?;
        let y = norm.forward(&x)?;
        assert_eq!(y.shape(), x.shape());
        assert!(y.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0);
        Ok(())
    }

    #[test]
    fn rmsnorm_forward_3d() -> Result<()> {
        let dev = Device::Cpu;
        let weight = Tensor::ones(&[4], DType::F32, &dev)?;
        let norm = RMSNorm::new(weight, 1e-5)?;
        // [B=2, T=3, D=4] — the old mean_keepdim(1) would
        // normalize over T instead of D, producing wrong results
        let x = Tensor::new(
            &[
                [
                    [1.0f32, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                ],
            ],
            &dev,
        )?;
        let y = norm.forward(&x)?;
        assert_eq!(y.shape(), x.shape());

        // Verify: each position-token is normalized independently.
        // So y[0,0,:] should only depend on x[0,0,:], not on x[0,1,:]
        // With old bug (mean_keepdim(1)), y[0,0] and y[0,1] would be the same
        // With correct fix, they should differ
        let y_slice = y.to_vec3::<f32>()?;
        // y[0,0,:] and y[0,1,:] MUST be different (different input, independent norm)
        assert_ne!(
            y_slice[0][0], y_slice[0][1],
            "3D RMSNorm: positions must be normalized independently"
        );
        // y[0,0,:] and y[1,0,:] should also differ
        assert_ne!(
            y_slice[0][0], y_slice[1][0],
            "3D RMSNorm: batches must differ"
        );
        Ok(())
    }
}
