//! LocEnc — Local Acoustic Encoder（feat_dim=64, patch_size=4）。
//!
//! 包含：
//! - `feat_encoder`：12 層 acoustic feature encoder
//! - `enc_to_lm_proj`：encoder → LM hidden 投影
//! - `feat_decoder.estimator`：12 層 decoder + DiT components
//! - `fsq_layer`：Finite Scalar Quantization
//! - `fusion_concat_proj`：fusion 投影
//! - `lm_to_dit_proj`、`res_to_dit_proj`：跨模態投影

use super::{attention::GQAAttention, mlp::MLP, rms_norm::RMSNorm, rope::RoPE};
use crate::config::EncoderConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
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
        let self_attn = GQAAttention::load(
            &pp,
            cfg.hidden_dim,
            cfg.num_heads,
            cfg.num_heads,
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
        let h = self.self_attn.forward(&h, rope, step)?;
        let h = (residual + h)?;
        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

/// LocEnc — 完整的 acoustic encoder-decoder 架構。
pub struct LocEnc {
    // feat_encoder
    pub encoder: Vec<EncLayer>,
    pub encoder_norm: RMSNorm,
    pub in_proj: candle_nn::Linear,
    pub special_token: Tensor,
    // projection
    pub enc_to_lm_proj: candle_nn::Linear,
    // feat_decoder (estimator)
    pub decoder: Vec<EncLayer>,
    pub decoder_norm: RMSNorm,
    pub cond_proj: candle_nn::Linear,
    pub time_mlp: (candle_nn::Linear, candle_nn::Linear),
    pub delta_time_mlp: (candle_nn::Linear, candle_nn::Linear),
    pub in_proj_dec: candle_nn::Linear,
    pub out_proj: candle_nn::Linear,
    // FSQ
    pub fsq_in_proj: candle_nn::Linear,
    pub fsq_out_proj: candle_nn::Linear,
    // fusion
    pub fusion_concat_proj: candle_nn::Linear,
    pub lm_to_dit_proj: candle_nn::Linear,
    pub res_to_dit_proj: candle_nn::Linear,
    // config
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
    ) -> Result<Self> {
        let dev = vb.device();
        let _ = DType::F32;

        // feat_encoder
        let mut encoder = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            encoder.push(EncLayer::load(vb, i, cfg, "feat_encoder.encoder", dev)?);
        }
        let encoder_norm = RMSNorm::load(vb, cfg.hidden_dim, 1e-5, "feat_encoder.encoder.norm")?;
        let in_proj =
            candle_nn::linear_no_bias(feat_dim, cfg.hidden_dim, vb.pp("feat_encoder.in_proj"))?;
        // special_token
        let special_token = vb.get(cfg.hidden_dim, "feat_encoder.special_token")?;

        let enc_to_lm_proj =
            candle_nn::linear_no_bias(cfg.hidden_dim, 2048, vb.pp("enc_to_lm_proj"))?;

        // feat_decoder.estimator
        let mut decoder = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            decoder.push(EncLayer::load(
                vb,
                i,
                cfg,
                "feat_decoder.estimator.decoder",
                dev,
            )?);
        }
        let decoder_norm = RMSNorm::load(
            vb,
            cfg.hidden_dim,
            1e-5,
            "feat_decoder.estimator.decoder.norm",
        )?;
        let cond_proj = candle_nn::linear(
            cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.cond_proj"),
        )?;
        let time_mlp_1 = candle_nn::linear(
            cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.time_mlp.linear_1"),
        )?;
        let time_mlp_2 = candle_nn::linear(
            cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.time_mlp.linear_2"),
        )?;
        let delta_time_mlp_1 = candle_nn::linear(
            cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.delta_time_mlp.linear_1"),
        )?;
        let delta_time_mlp_2 = candle_nn::linear(
            cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.delta_time_mlp.linear_2"),
        )?;
        let in_proj_dec = candle_nn::linear(
            cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.in_proj"),
        )?;
        let out_proj = candle_nn::linear(
            cfg.hidden_dim,
            feat_dim,
            vb.pp("feat_decoder.estimator.out_proj"),
        )?;

        // FSQ
        let fsq_in_proj =
            candle_nn::linear_no_bias(cfg.hidden_dim, 512, vb.pp("fsq_layer.in_proj"))?;
        let fsq_out_proj =
            candle_nn::linear_no_bias(512, cfg.hidden_dim, vb.pp("fsq_layer.out_proj"))?;

        // fusion
        let fusion_concat_proj = candle_nn::linear_no_bias(
            2048 + cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("fusion_concat_proj"),
        )?;
        let lm_to_dit_proj =
            candle_nn::linear_no_bias(2048, cfg.hidden_dim, vb.pp("lm_to_dit_proj"))?;
        let res_to_dit_proj =
            candle_nn::linear_no_bias(2048, cfg.hidden_dim, vb.pp("res_to_dit_proj"))?;

        // RoPE for encoder/decoder (use standard RoPE)
        let rope = RoPE::new(8192, cfg.kv_channels, 10000.0, None, dev)?;

        Ok(Self {
            encoder,
            encoder_norm,
            in_proj,
            special_token,
            enc_to_lm_proj,
            decoder,
            decoder_norm,
            cond_proj,
            time_mlp: (time_mlp_1, time_mlp_2),
            delta_time_mlp: (delta_time_mlp_1, delta_time_mlp_2),
            in_proj_dec,
            out_proj,
            fsq_in_proj,
            fsq_out_proj,
            fusion_concat_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            feat_dim,
            patch_size,
            hidden_dim: cfg.hidden_dim,
            rope,
        })
    }

    /// Encode acoustic features to LM hidden states.
    pub fn encode(&mut self, acoustic_features: &Tensor) -> Result<Tensor> {
        // acoustic_features: [batch, feat_dim, time]
        // → [batch, time, feat_dim]
        let x = acoustic_features.transpose(1, 2)?;
        let x = self.in_proj.forward(&x)?; // → [batch, time, hidden_dim]
        let mut h = x.clone();
        for layer in self.encoder.iter_mut() {
            h = layer.forward(&h, &self.rope, 0)?;
        }
        h = self.encoder_norm.forward(&h)?;
        self.enc_to_lm_proj.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn locenc_shape_test() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = EncoderConfig {
            hidden_dim: 1024,
            ffn_dim: 4096,
            num_heads: 16,
            num_layers: 2,
            kv_channels: 128,
        };

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let mut loc = LocEnc::load(&vb, &cfg, 64, 4)?;
        let feats = Tensor::zeros(&[1, 64, 10], DType::F32, &dev)?;
        let out = loc.encode(&feats)?;
        assert_eq!(out.dims(), &[1, 10, 2048]);
        Ok(())
    }
}
