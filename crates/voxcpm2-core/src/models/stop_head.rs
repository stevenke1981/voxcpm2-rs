//! Stop Head — binary classifier for early-stop prediction in autoregressive loop.
//!
//! Tensor structure (from `model.safetensors`):
//!
//! stop_proj.weight: [2048, 2048]    Linear(2048, 2048) with bias
//! stop_proj.bias:   [2048]
//! stop_head.weight: [2, 2048]       Linear(2048, 2) without bias!
//!
//! Note: stop_head has NO bias (confirmed from safetensors).
//! No weights for stop_actn (SiLU activation).

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Stop predictor: projects LM hidden state to a binary stop/continue decision.
pub struct StopHead {
    stop_proj: candle_nn::Linear,
    stop_head: candle_nn::Linear,
}

impl StopHead {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let stop_proj = candle_nn::linear(2048, 2048, vb.pp("stop_proj"))?;
        let stop_head = candle_nn::linear_no_bias(2048, 2, vb.pp("stop_head"))?;
        Ok(Self {
            stop_proj,
            stop_head,
        })
    }

    /// Forward pass: [B, T, 2048] → [B, T, 2] (stop logits).
    ///
    /// Python equivalent:
    ///   x = stop_proj(hidden)  # [B, T, 2048]
    ///   x = SiLU(x)            # in-place activation
    ///   x = stop_head(x)       # [B, T, 2]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.stop_proj.forward(x)?; // [B, T, 2048]
        let h = candle_nn::ops::silu(&h)?; // [B, T, 2048]
        self.stop_head.forward(&h) // [B, T, 2]
    }

    /// Convenience: check whether to stop given the logits.
    ///
    /// Returns `true` if stop_prob > 0.5 (i.e. logit[1] > logit[0]).
    pub fn should_stop(&self, logits: &Tensor) -> Result<bool> {
        // Take last position and compute argmax
        let last = logits.narrow(1, logits.dim(1)? - 1, 1)?; // [B, 1, 2]
        let last = last.squeeze(1)?; // [B, 2]
                                     // softmax to get probabilities
        let probs = candle_nn::ops::softmax(&last, 1)?;
        let p_stop = probs
            .narrow(1, 1, 1)?
            .squeeze(1)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        Ok(p_stop[0] > 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn stop_head_shape_test() -> Result<()> {
        let dev = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let stop = StopHead::load(&vb)?;
        let x = Tensor::zeros(&[1, 5, 2048], DType::F32, &dev)?;
        let y = stop.forward(&x)?;
        assert_eq!(y.dims(), &[1, 5, 2], "StopHead forward shape");
        Ok(())
    }

    #[test]
    fn should_stop_false_by_default() -> Result<()> {
        let dev = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let stop = StopHead::load(&vb)?;
        // With zero weights, both logits are 0, so softmax gives [0.5, 0.5]
        // should_stop returns p_stop > 0.5 = false
        let x = Tensor::zeros(&[1, 3, 2048], DType::F32, &dev)?;
        let logits = stop.forward(&x)?;
        let result = stop.should_stop(&logits)?;
        assert!(!result, "with zero weights should not stop");
        Ok(())
    }
}
