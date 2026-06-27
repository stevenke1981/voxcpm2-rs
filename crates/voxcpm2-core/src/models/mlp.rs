//! SwiGLU MLP — 用於所有 transformer 子層。
//!
//! 對應 `base_lm.layers.*.mlp.gate_proj / up_proj / down_proj`。
//! 公式: `down_proj(silu(gate_proj(x)) * up_proj(x))`

use candle_core::{Module, Result};
use candle_nn::{Activation, VarBuilder};

#[derive(Debug, Clone)]
pub struct MLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl MLP {
    /// 建立 MLP（hidden_size → intermediate_size → hidden_size）。
    pub fn load(
        vb: &VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        prefix: &str,
    ) -> Result<Self> {
        let gate_proj = candle_nn::linear_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp(format!("{prefix}.gate_proj")),
        )?;
        let up_proj = candle_nn::linear_no_bias(
            hidden_size,
            intermediate_size,
            vb.pp(format!("{prefix}.up_proj")),
        )?;
        let down_proj = candle_nn::linear_no_bias(
            intermediate_size,
            hidden_size,
            vb.pp(format!("{prefix}.down_proj")),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = Activation::Silu.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn mlp_forward_shape() -> Result<()> {
        let dev = Device::Cpu;
        // Use dummy constructors: for shape test we skip weight loading
        let lin = |in_dim: usize, out_dim: usize| {
            let w = Tensor::zeros(&[out_dim, in_dim], candle_core::DType::F32, &dev).unwrap();
            candle_nn::Linear::new(w, None)
        };
        let mlp = MLP {
            gate_proj: lin(4, 16),
            up_proj: lin(4, 16),
            down_proj: lin(16, 4),
        };
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &dev)?;
        let y = mlp.forward(&x)?;
        assert_eq!(y.dims(), &[1, 4]);
        Ok(())
    }
}
