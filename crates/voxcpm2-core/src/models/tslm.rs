//! TSLM — Text-Shaped Language Model backbone（28 層 MiniCPM-like）。
//!
//! 對應 tensor name prefix: `base_lm.*`

use super::{attention::GQAAttention, mlp::MLP, rms_norm::RMSNorm, rope::RoPE};
use crate::config::LmConfig;
#[cfg(test)]
use candle_core::DType;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// 單一 TSLM Transformer 層。
pub struct TslmLayer {
    input_layernorm: RMSNorm,
    self_attn: GQAAttention,
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
}

impl TslmLayer {
    /// 從 VarBuilder 載入第 i 層權重。
    pub fn load(
        vb: &VarBuilder,
        i: usize,
        cfg: &LmConfig,
        use_kv_cache: bool,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        // vb already has "base_lm." prefix (applied by caller)
        let prefix = format!("layers.{i}");
        let pp = vb.pp(&prefix);
        let eps = cfg.rms_norm_eps;

        let input_layernorm = RMSNorm::load(&pp, cfg.hidden_size, eps, "input_layernorm")?;
        let self_attn = GQAAttention::load(
            &pp,
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.kv_channels,
            "self_attn",
            use_kv_cache,
            max_seq_len,
            dev,
        )?;
        let post_attention_layernorm =
            RMSNorm::load(&pp, cfg.hidden_size, eps, "post_attention_layernorm")?;
        let mlp = MLP::load(&pp, cfg.hidden_size, cfg.intermediate_size, "mlp")?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(&mut self, x: &Tensor, rope: &RoPE, step: usize) -> Result<Tensor> {
        // Pre-attention norm
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, rope, step)?;
        let h = (residual + h)?;

        // Pre-MLP norm
        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = (residual + h)?;
        Ok(h)
    }
}

/// TSLM 完整模型（28 層 + embedding + final norm）。
pub struct TSLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<TslmLayer>,
    norm: RMSNorm,
    rope: RoPE,
    scale_emb: f64,
    scale_depth: f64,
}

impl TSLM {
    pub fn load(vb: &VarBuilder, cfg: &LmConfig, dev: &Device, use_kv_cache: bool) -> Result<Self> {
        // LongRoPE factors
        let factors: Option<Vec<f64>> = cfg.rope_scaling.as_ref().map(|rs| rs.long_factor.clone());
        let factors_ref = factors.as_deref();

        let rope = RoPE::new(
            cfg.max_position_embeddings,
            cfg.kv_channels,
            cfg.rope_theta,
            factors_ref,
            dev,
        )?;

        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embed_tokens"),
        )?;
        let norm = RMSNorm::load(vb, cfg.hidden_size, cfg.rms_norm_eps, "norm")?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TslmLayer::load(
                vb,
                i,
                cfg,
                use_kv_cache,
                cfg.max_position_embeddings,
                dev,
            )?);
        }

        let scale_emb = cfg.scale_emb.unwrap_or(1.0);
        let scale_depth = cfg.scale_depth.unwrap_or(1.0);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            scale_emb,
            scale_depth,
        })
    }

    /// Forward: input_ids shape [batch, seq_len]
    /// 回傳 hidden states: [batch, seq_len, hidden_size]
    pub fn forward(&mut self, input_ids: &Tensor, step: usize) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        // muP scaling
        if self.scale_emb != 1.0 {
            h = (h * self.scale_emb)?;
        }

        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, &self.rope, step)?;
        }

        h = self.norm.forward(&h)?;
        // muP scale_depth
        if self.scale_depth != 1.0 {
            h = (h / self.scale_depth)?;
        }
        Ok(h)
    }

    /// LM head: 投影 hidden 到 vocab logits。
    /// 權重綁定（tie weights）— 與 embed_tokens 共享。
    pub fn lm_head(&self, hidden: &Tensor, tied_weight: &Tensor) -> Result<Tensor> {
        hidden.matmul(&tied_weight.t()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn tslm_construction() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = LmConfig {
            bos_token_id: 1,
            eos_token_id: 73440,
            hidden_size: 2048,
            intermediate_size: 6144,
            max_position_embeddings: 64,
            num_attention_heads: 16,
            num_hidden_layers: 2,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            kv_channels: 128,
            vocab_size: 100,
            use_mup: false,
            scale_emb: Some(12.0),
            dim_model_base: Some(256.0),
            scale_depth: Some(1.4),
        };

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);
        let _tslm = TSLM::load(&vb, &cfg, &dev, false)?;
        // Construction succeeds — forward requires CUDA (index_select F32 on CPU unsupported)
        Ok(())
    }
}
