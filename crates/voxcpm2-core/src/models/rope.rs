//! RoPE / LongRoPE 位置編碼。
//!
//! VoxCPM2 使用 LongRoPE，具有 long_factor 與 short_factor 兩組
//! 64 維的頻率縮放因子。本實作支援：
//! - 標準 RoPE（使用 theta）
//! - LongRoPE（使用 factor 陣列，分長/短兩種）
//! - 可選 `residual_lm_no_rope`（RALM 不使用 RoPE）

use candle_core::{DType, Device, Result, Tensor};

/// 預計算 cos/sin 表格，支援 LongRoPE 縮放。
pub struct RoPE {
    sin: Tensor,
    cos: Tensor,
}

impl RoPE {
    /// 建立 RoPE。
    ///
    /// - `seq_len`：最大序列長度（用於預計算表格）
    /// - `head_dim`：每個注意力頭的維度（= kv_channels = 128）
    /// - `theta`：base frequency（預設 10000.0）
    /// - `factors`：可選的 LongRoPE 縮放因子（長度 = head_dim/2 = 64）
    pub fn new(
        seq_len: usize,
        head_dim: usize,
        theta: f64,
        factors: Option<&[f64]>,
        dev: &Device,
    ) -> Result<Self> {
        let half = head_dim / 2;
        // 頻率計算: inv_freq[i] = 1.0 / (theta^(2i/head_dim))
        let mut inv_freq: Vec<f64> = Vec::with_capacity(half);
        for i in 0..half {
            let val = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            inv_freq.push(val);
        }

        // LongRoPE 縮放
        if let Some(factors) = factors {
            assert_eq!(
                factors.len(),
                half,
                "LongRoPE factor length must equal head_dim/2"
            );
            for (f, factor) in inv_freq.iter_mut().zip(factors) {
                *f *= factor;
            }
        }

        let inv_freq = Tensor::from_vec(inv_freq, half, dev)?.to_dtype(DType::F32)?;
        // shape: [half]
        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, seq_len, dev)?;
        // shape: [seq_len, half]
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?.to_dtype(DType::F32)?;
        let sin = freqs.sin()?.to_dtype(DType::F32)?;

        Ok(Self { sin, cos })
    }

    /// 應用 RoPE 到 q 和 k。
    /// q/k shape: [batch, seq_len, num_heads, head_dim]
    /// 回傳: (q_rotated, k_rotated)
    pub fn apply(&self, q: &Tensor, k: &Tensor, seq_len_offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos.narrow(0, seq_len_offset, seq_len)?;
        let sin = self.sin.narrow(0, seq_len_offset, seq_len)?;

        let q_rot = self.apply_rotary(q, &cos, &sin)?;
        let k_rot = self.apply_rotary(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    /// 對單一張量應用旋轉。
    fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = x.dim(3)? / 2;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;
        // 旋轉: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        // cos/sin shape: [seq_len, half] → broadcast to [1, seq_len, 1, half]
        let cos_b = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin_b = sin.unsqueeze(0)?.unsqueeze(2)?;
        let rotated1 = (x1.broadcast_mul(&cos_b)? - x2.broadcast_mul(&sin_b)?)?;
        let rotated2 = (x1.broadcast_mul(&sin_b)? + x2.broadcast_mul(&cos_b)?)?;
        Tensor::cat(&[&rotated1, &rotated2], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_basic_shape() -> Result<()> {
        let dev = Device::Cpu;
        let rope = RoPE::new(10, 128, 10000.0, None, &dev)?;
        let q = Tensor::zeros(&[1, 5, 16, 128], DType::F32, &dev)?;
        let k = Tensor::zeros(&[1, 5, 2, 128], DType::F32, &dev)?;
        let (q2, k2) = rope.apply(&q, &k, 0)?;
        assert_eq!(q2.shape(), q.shape());
        assert_eq!(k2.shape(), k.shape());
        Ok(())
    }
}
