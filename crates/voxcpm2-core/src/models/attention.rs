//! GQA (Grouped Query Attention) 注意力層。
//!
//! VoxCPM2 使用 GQA：num_attention_heads=16, num_key_value_heads=2。
//! 每個 KV head 服務 8 個 query head。
//!
//! 支援 KV cache 以實現 autoregressive decode。

#[cfg(test)]
use candle_core::DType;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::rope::RoPE;

#[derive(Debug, Clone, Default)]
pub struct KVCache {
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor, _step: usize) -> Result<(Tensor, Tensor)> {
        let k_new = match &self.k_cache {
            Some(prev) => Tensor::cat(&[prev, k], 1)?,
            None => k.clone(),
        };
        let v_new = match &self.v_cache {
            Some(prev) => Tensor::cat(&[prev, v], 1)?,
            None => v.clone(),
        };
        self.k_cache = Some(k_new.clone());
        self.v_cache = Some(v_new.clone());
        Ok((k_new, v_new))
    }
}

#[derive(Debug, Clone)]
pub struct GQAAttention {
    pub q_proj: candle_nn::Linear,
    pub k_proj: candle_nn::Linear,
    pub v_proj: candle_nn::Linear,
    pub o_proj: candle_nn::Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_cache: Option<KVCache>,
}

impl GQAAttention {
    /// 從 VarBuilder 載入權重。
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        kv_channels: usize,
        prefix: &str,
        use_kv_cache: bool,
        _max_seq_len: usize,
        _dev: &Device,
    ) -> Result<Self> {
        let head_dim = kv_channels;
        let q_proj = candle_nn::linear_no_bias(
            hidden_size,
            num_heads * head_dim,
            vb.pp(format!("{prefix}.q_proj")),
        )?;
        let k_proj = candle_nn::linear_no_bias(
            hidden_size,
            num_kv_heads * head_dim,
            vb.pp(format!("{prefix}.k_proj")),
        )?;
        let v_proj = candle_nn::linear_no_bias(
            hidden_size,
            num_kv_heads * head_dim,
            vb.pp(format!("{prefix}.v_proj")),
        )?;
        let o_proj = candle_nn::linear_no_bias(
            num_heads * head_dim,
            hidden_size,
            vb.pp(format!("{prefix}.o_proj")),
        )?;

        let kv_cache = if use_kv_cache {
            Some(KVCache::new())
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Forward: 輸入 x shape [batch, seq_len, hidden_size]
    /// 回傳: [batch, seq_len, hidden_size]
    pub fn forward(&mut self, x: &Tensor, rope: &RoPE, step: usize) -> Result<Tensor> {
        let (b, seq_len, _) = x.shape().dims3()?;

        // Projections
        let q = self.q_proj.forward(x)?; // [b, seq_len, num_heads * head_dim]
        let k = self.k_proj.forward(x)?; // [b, seq_len, num_kv_heads * head_dim]
        let v = self.v_proj.forward(x)?; // [b, seq_len, num_kv_heads * head_dim]

        // Reshape to [b, seq_len, num_heads, head_dim] / [b, seq_len, num_kv_heads, head_dim]
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;

        // RoPE
        let (q, k) = rope.apply(&q, &k, step)?;

        // KV cache
        let (k, v) = if let Some(cache) = &mut self.kv_cache {
            cache.append(&k, &v, step)?
        } else {
            (k, v)
        };
        let full_seq_len = k.dim(1)?;

        // GQA: expand KV heads to match Q heads
        // q: [b, seq_len, num_heads, head_dim]
        // k: [b, full_seq_len, num_kv_heads, head_dim]
        let group_size = self.num_heads / self.num_kv_heads;
        // Expand k/v: [b, full_seq_len, num_kv_heads, head_dim] → [b, full_seq_len, num_heads, head_dim]
        let k = k.unsqueeze(3)?.expand((
            b,
            full_seq_len,
            self.num_kv_heads,
            group_size,
            self.head_dim,
        ))?;
        let k = k.reshape((b, full_seq_len, self.num_heads, self.head_dim))?;
        let v = v.unsqueeze(3)?.expand((
            b,
            full_seq_len,
            self.num_kv_heads,
            group_size,
            self.head_dim,
        ))?;
        let v = v.reshape((b, full_seq_len, self.num_heads, self.head_dim))?;

        // Scaled dot-product attention
        // q, k, v all: [b, seq_len/full_seq_len, num_heads, head_dim]
        // Transpose to [b, num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt().recip();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;
        let attn_output = attn_weights.matmul(&v)?; // [b, num_heads, seq_len, head_dim]

        // Transpose back and reshape
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::rope::RoPE;
    use candle_core::Device;

    #[test]
    fn gqa_forward_shape() -> Result<()> {
        let dev = Device::Cpu;
        // GQA: 16 heads, 2 KV heads, head_dim=128, hidden=2048
        let hidden = 2048;
        let head_dim = 128;
        let num_heads = 16;
        let num_kv_heads = 2;
        let seq_len = 4;
        let rope = RoPE::new(64, head_dim, 10000.0, None, &dev)?;

        // Build with zero weights (for shape test only)
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let mut attn = GQAAttention::load(
            &vb,
            hidden,
            num_heads,
            num_kv_heads,
            head_dim,
            "attn",
            false,
            0,
            &dev,
        )?;
        let x = Tensor::zeros(&[1, seq_len, hidden], DType::F32, &dev)?;
        let y = attn.forward(&x, &rope, 0)?;
        assert_eq!(y.dims(), &[1, seq_len, hidden]);
        Ok(())
    }
}
