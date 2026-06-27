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

    /// forward: `x * rsqrt(mean(x^2) + eps) * weight`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f64 = x.to_dtype(DType::F64)?;
        let norm = x_f64.sqr()?.mean_keepdim(1)?;
        let x_normed = x_f64.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x.dtype())?;
        x_normed.broadcast_mul(&self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_forward() -> Result<()> {
        let dev = Device::Cpu;
        let weight = Tensor::ones(&[4], DType::F32, &dev)?;
        let norm = RMSNorm::new(weight, 1e-5)?;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &dev)?;
        let y = norm.forward(&x)?;
        assert_eq!(y.shape(), x.shape());
        assert!(y.abs()?.sum_all()?.to_scalar::<f32>()? > 0.0);
        Ok(())
    }
}
