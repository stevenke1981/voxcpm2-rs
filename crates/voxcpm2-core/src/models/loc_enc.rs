//! LocEnc — Local Acoustic Encoder（feat_dim=64, patch_size=4）。
//!
//! 僅包含 feat_encoder 部分（encoder layers + enc_to_lm_proj）。
//! DiT/FSQ/fusion 由 `LocDiT`、`FsqLayer`、`autoregressive.rs` 單獨載入。

use super::{attention::GQAAttention, mlp::MLP, rms_norm::RMSNorm, rope::RoPE};
use crate::config::EncoderConfig;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// 編碼器層（無 RoPE — LocEnc 依賴位置由 patch embedding 提供）。
pub struct EncLayer {
    input_layernorm: RMSNorm,
    self_attn: GQAAttention,
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
}

impl EncLayer {
    pub fn load(
        vb: &VarBuilder,
        i: usize,
        cfg: &EncoderConfig,
        prefix: &str,
        dev: &Device,
    ) -> Result<Self> {
        let pp = vb.pp(format!("{prefix}.layers.{i}"));
        let eps = 1e-5;
        let input_layernorm = RMSNorm::load(&pp, cfg.hidden_dim, eps, "input_layernorm")?;
        // GQA ratio 8:1 (16 heads, 2 kv_heads for VoxCPM2 encoder/decoder)
        let num_kv_heads = (cfg.num_heads / 8).max(1);
        let self_attn = GQAAttention::load(
            &pp,
            cfg.hidden_dim,
            cfg.num_heads,
            num_kv_heads,
            cfg.kv_channels,
            "self_attn",
            false,
            0,
            dev,
        )?;
        let post_attention_layernorm =
            RMSNorm::load(&pp, cfg.hidden_dim, eps, "post_attention_layernorm")?;
        let mlp = MLP::load(&pp, cfg.hidden_dim, cfg.ffn_dim, "mlp")?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(&mut self, x: &Tensor, rope: &RoPE, step: usize) -> Result<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, rope, step, false)?;
        let h = (residual + h)?;
        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

/// LocEnc — feat_encoder 部分（encoder layers + enc_to_lm_proj）。
pub struct LocEnc {
    pub encoder: Vec<EncLayer>,
    pub encoder_norm: RMSNorm,
    pub in_proj: candle_nn::Linear,
    pub special_token: Tensor,
    pub enc_to_lm_proj: candle_nn::Linear,
    pub feat_dim: usize,
    pub patch_size: usize,
    pub hidden_dim: usize,
    pub rope: RoPE,
}

impl LocEnc {
    pub fn load(
        vb: &VarBuilder,
        cfg: &EncoderConfig,
        feat_dim: usize,
        patch_size: usize,
        rope_factors: Option<&[f64]>,
    ) -> Result<Self> {
        let dev = vb.device();

        // feat_encoder (16 heads, 2 kv_heads = 8:1 GQA ratio in real model)
        // For test configs with fewer heads, min 1
        let num_kv_heads = (cfg.num_heads / 8).max(1);
        let mut encoder = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let pp = vb.pp(format!("feat_encoder.encoder.layers.{i}"));
            let eps = 1e-5;
            let input_layernorm = RMSNorm::load(&pp, cfg.hidden_dim, eps, "input_layernorm")?;
            let self_attn = GQAAttention::load(
                &pp,
                cfg.hidden_dim,
                cfg.num_heads,
                num_kv_heads,
                cfg.kv_channels,
                "self_attn",
                false,
                0,
                dev,
            )?;
            let post_attention_layernorm =
                RMSNorm::load(&pp, cfg.hidden_dim, eps, "post_attention_layernorm")?;
            let mlp = MLP::load(&pp, cfg.hidden_dim, cfg.ffn_dim, "mlp")?;
            encoder.push(EncLayer {
                input_layernorm,
                self_attn,
                post_attention_layernorm,
                mlp,
            });
        }
        let encoder_norm = RMSNorm::load(vb, cfg.hidden_dim, 1e-5, "feat_encoder.encoder.norm")?;
        let in_proj =
            candle_nn::linear(feat_dim, cfg.hidden_dim, vb.pp("feat_encoder.in_proj"))?;
        // special_token: [1, 1, 1, hidden_dim] → squeeze down to [1, 1, hidden_dim]
        let special_token_4d = vb.get(&[1, 1, 1, cfg.hidden_dim], "feat_encoder.special_token")?;
        let special_token = special_token_4d.squeeze(2)?; // [1, 1, hidden_dim]

        let enc_to_lm_proj =
            candle_nn::linear(cfg.hidden_dim, 2048, vb.pp("enc_to_lm_proj"))?; // has bias in safetensors

        // RoPE for encoder (use short_factor from lm_config, same as DiT decoder)
        let rope = RoPE::new(8192, cfg.kv_channels, 10000.0, rope_factors, dev)?;

        Ok(Self {
            encoder,
            encoder_norm,
            in_proj,
            special_token,
            enc_to_lm_proj,
            feat_dim,
            patch_size,
            hidden_dim: cfg.hidden_dim,
            rope,
        })
    }

    /// Encode acoustic features to LM hidden state.
    ///
    /// 對應 Python `VoxCPMLocEnc.forward()`:
    /// - Python 輸入: `[B, T=1, P=patch_size=4, D=64]` (4D)
    /// - Rust 輸入: `[B, C=64, T=patch_size=4]` (channels-first 3D)
    /// - reshape → `[B, T, C]` → `in_proj` → `[B, T, hidden]`
    /// - Python: `special_token` (learned `[1,1,1,hidden]`) placed at **position 0**,
    ///   then `x` features at positions 1..P.
    /// - rearrage `(b t) p c` → `[B, 1+P, hidden]` (T=1 per-step)
    /// - encoder forward (bidirectional)
    /// - 取**第一個位置**（special token position 0）→ `[B, 1, hidden]`
    /// - `enc_to_lm_proj` → `[B, 1, 2048]`
    pub fn encode(&mut self, acoustic_features: &Tensor) -> Result<Tensor> {
        let x = acoustic_features.transpose(1, 2)?;    // [B, C, T] → [B, T, C]
        let x = self.in_proj.forward(&x)?;               // [B, T, hidden]
        let batch = x.dim(0)?;
        // Python: special_token is [1, 1, 1, hidden] → expand to [B, T=1, 1, hidden]
        // Then cat([special, x], dim=2) → [B, 1, 1+P, hidden]
        // Then rearrange → [B*1, 1+P, hidden] → same as [B, 1+P, hidden]
        // Our special_token is [1, 1, hidden]; expand to [batch, 1, hidden]
        let special = self.special_token.expand(&[batch, 1, self.hidden_dim])?;
        // !!! CRITICAL: place special FIRST (position 0), THEN features (positions 1..P)
        // to match Python's ordering and RoPE positions.
        let x = Tensor::cat(&[&special, &x], 1)?;        // [B, 1+P, hidden]
        let mut h = x;
        for layer in self.encoder.iter_mut() {
            h = layer.forward(&h, &self.rope, 0)?;
        }
        h = self.encoder_norm.forward(&h)?;
        // Python: cls_output = outputs[:, 0, :] → take position 0 (special token)
        let h = h.narrow(1, 0, 1)?;                      // [B, 1, hidden]
        self.enc_to_lm_proj.forward(&h)                   // [B, 1, 2048]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn locenc_shape_test() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = EncoderConfig {
            hidden_dim: 256,
            ffn_dim: 1024,
            num_heads: 4,
            num_layers: 2,
            kv_channels: 64,
        };

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let mut loc = LocEnc::load(&vb, &cfg, 64, 4, None)?;
        // Simulate one predicted patch: [B, C=64, T=4]
        let feats = Tensor::zeros(&[1, 64, 4], DType::F32, &dev)?;
        let out = loc.encode(&feats)?;
        // Output should be [B, 1, 2048] (single vector per patch)
        assert_eq!(out.dims(), &[1, 1, 2048], "encode: [B, 1, 2048]");
        Ok(())
    }
}
