//! Feature Encoder — 12-layer GQA Transformer that maps 64-dim acoustic features
//! to 1024-dim text conditioning space.
//!
//! Tensor structure (from model.safetensors `feat_encoder.*`):
//!
//! feat_encoder.in_proj.weight: [1024, 64]    Linear(64, 1024)
//! feat_encoder.in_proj.bias:   [1024]
//! feat_encoder.encoder.norm.weight: [1024]   RMSNorm(1024)
//! feat_encoder.special_token:  [1, 1, 1, 1024]
//!
//! feat_encoder.encoder.layers.N (N=0..11):
//!   input_layernorm.weight: [1024]              RMSNorm(1024)
//!   self_attn.q_proj.weight: [2048, 1024]       Linear(1024, 2048)
//!   self_attn.k_proj.weight: [256, 1024]         Linear(1024, 256)
//!   self_attn.v_proj.weight: [256, 1024]         Linear(1024, 256)
//!   self_attn.o_proj.weight: [1024, 2048]        Linear(2048, 1024)
//!   post_attention_layernorm.weight: [1024]      RMSNorm(1024)
//!   mlp.gate_proj.weight: [4096, 1024]           Linear(1024, 4096)
//!   mlp.up_proj.weight:   [4096, 1024]           Linear(1024, 4096)
//!   mlp.down_proj.weight: [1024, 4096]           Linear(4096, 1024)

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::{GQAAttention, RMSNorm, MLP};
use crate::config::EncoderConfig;

/// A single Transformer encoder layer with GQA self-attention + SwiGLU MLP.
#[derive(Debug, Clone)]
pub struct FeatEncoderLayer {
    input_layernorm: RMSNorm,
    self_attn: GQAAttention,
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
}

impl FeatEncoderLayer {
    pub fn load(
        vb: &VarBuilder,
        prefix: &str,
        cfg: &EncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.kv_channels;
        let input_layernorm = RMSNorm::load(
            vb,
            cfg.hidden_dim,
            1e-5,
            &format!("{prefix}.input_layernorm"),
        )?;

        // GQA: feat_encoder weights show:
        //   q_proj: [2048, 1024] → output = 2048 = 16 heads × 128
        //   k_proj: [256, 1024]  → output = 256  = 2  heads × 128
        // So num_heads=16, num_kv_heads=2. The ratio is 8:1.
        // We compute kv_heads from the total-kv-dim / kv_channels ratio.
        // q_proj dim = num_heads * kv_channels.
        // k_proj dim = num_kv_heads * kv_channels (num_kv_heads inferred).
        let num_kv_heads = cfg.num_heads / 8; // 16/8 = 2 for VoxCPM2 feat_encoder
        let self_attn = GQAAttention::load(
            vb,
            cfg.hidden_dim,
            cfg.num_heads,
            num_kv_heads,
            head_dim,
            &format!("{prefix}.self_attn"),
            false, // no KV cache
            0,
            device,
        )?;

        let post_attention_layernorm = RMSNorm::load(
            vb,
            cfg.hidden_dim,
            1e-5,
            &format!("{prefix}.post_attention_layernorm"),
        )?;

        let mlp = MLP::load(vb, cfg.hidden_dim, cfg.ffn_dim, &format!("{prefix}.mlp"))?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(&mut self, x: &Tensor, device: &Device) -> Result<Tensor> {
        // Pre-norm: x → norm → attn → residual
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        // GQAAttention::forward requires RoPE & step; for non-RoPE use a dummy
        let h = self.self_attn_forward_no_rope(&h, device)?;
        let h = (h + residual)?;

        // Pre-norm: h → norm → mlp → residual
        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        h + residual
    }

    /// GQA forward without RoPE (feat_encoder doesn't use positional encoding).
    fn self_attn_forward_no_rope(&mut self, x: &Tensor, _device: &Device) -> Result<Tensor> {
        // Use a simple self-attention path without RoPE
        let (b, seq_len, _) = x.shape().dims3()?;

        let q = self.self_attn.q_proj.forward(x)?; // [b, seq, num_heads * head_dim]
        let k = self.self_attn.k_proj.forward(x)?;
        let v = self.self_attn.v_proj.forward(x)?;

        let head_dim = self.self_attn.head_dim();
        let num_heads = self.self_attn.num_heads();
        let num_kv_heads = self.self_attn.num_kv_heads();

        let q = q.reshape((b, seq_len, num_heads, head_dim))?;
        let k = k.reshape((b, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((b, seq_len, num_kv_heads, head_dim))?;

        // Expand KV heads to match Q heads (GQA)
        let group_size = num_heads / num_kv_heads;
        let k = k
            .unsqueeze(3)?
            .expand((b, seq_len, num_kv_heads, group_size, head_dim))?
            .reshape((b, seq_len, num_heads, head_dim))?;
        let v = v
            .unsqueeze(3)?
            .expand((b, seq_len, num_kv_heads, group_size, head_dim))?
            .reshape((b, seq_len, num_heads, head_dim))?;

        // Transpose to [b, num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let scale = (head_dim as f64).sqrt().recip();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Transpose back
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, seq_len, num_heads * head_dim))?;

        self.self_attn.o_proj.forward(&attn_output)
    }
}

/// Feature Encoder: 64-dim → in_proj → 12×Transformer → norm → 1024-dim
pub struct FeatEncoder {
    in_proj: candle_nn::Linear,
    layers: Vec<FeatEncoderLayer>,
    norm: RMSNorm,
    special_token: Tensor, // [1, 1, 1, 1024]
    pub feat_dim: usize,
    pub hidden_dim: usize,
}

impl FeatEncoder {
    pub fn load(vb: &VarBuilder, cfg: &EncoderConfig, device: &Device) -> Result<Self> {
        let feat_dim = 64; // hard-coded from VoxCPM2 architecture
        let hidden_dim = cfg.hidden_dim;

        let in_proj = candle_nn::linear(feat_dim, hidden_dim, vb.pp("feat_encoder.in_proj"))?;

        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            let layer = FeatEncoderLayer::load(
                vb,
                &format!("feat_encoder.encoder.layers.{i}"),
                cfg,
                device,
            )?;
            layers.push(layer);
        }

        let norm = RMSNorm::load(vb, hidden_dim, 1e-5, "feat_encoder.encoder.norm")?;

        // special_token: [1, 1, 1, 1024]
        let special_token = vb.get(&[1, 1, 1, hidden_dim], "feat_encoder.special_token")?;

        Ok(Self {
            in_proj,
            layers,
            norm,
            special_token,
            feat_dim,
            hidden_dim,
        })
    }

    /// Forward: 64-dim input → 1024-dim output with self-attention.
    ///
    /// # Args
    /// * `x` — shape `[batch, seq_len, feat_dim=64]`
    ///
    /// # Returns
    /// * shape `[batch, seq_len, hidden_dim=1024]`
    pub fn forward(&mut self, x: &Tensor, device: &Device) -> Result<Tensor> {
        let (_b, _seq_len, _fd) = x.shape().dims3()?;

        // Project 64 → 1024
        let mut h = self.in_proj.forward(x)?; // [b, S, 1024]

        // Add special token (broadcast from [1,1,1,1024] → [b, S, 1024])
        let tok = self.special_token.squeeze(0)?.squeeze(0)?; // [1024]
        let tok = tok.reshape((1, 1, self.hidden_dim))?; // [1, 1, 1024]
        h = h.broadcast_add(&tok)?;

        // 12 transformer layers
        for layer in &mut self.layers {
            h = layer.forward(&h, device)?;
        }

        // Final norm
        self.norm.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_core::Device;

    #[test]
    fn feat_encoder_shape_test() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = EncoderConfig {
            hidden_dim: 1024,
            ffn_dim: 4096,
            num_heads: 16,
            num_layers: 12,
            kv_channels: 128,
        };

        // Build with zero weights
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);
        let mut enc = FeatEncoder::load(&vb, &cfg, &dev)?;

        // Input: [1, 4, 64]
        let x = Tensor::zeros(&[1, 4, 64], DType::F32, &dev)?;
        let y = enc.forward(&x, &dev)?;
        assert_eq!(y.dims(), &[1, 4, 1024]);
        Ok(())
    }
}
