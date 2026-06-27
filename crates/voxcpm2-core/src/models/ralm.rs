//! RALM — Residual Acoustic Language Model（8 層，不含 RoPE）。
//!
//! 對應 tensor name prefix: `residual_lm.*`
//!
//! VoxCPM2 中 `residual_lm_no_rope = true`，所以 RALM 不使用位置編碼。

use super::{attention::GQAAttention, mlp::MLP, rms_norm::RMSNorm};
use crate::config::LmConfig;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// 單一 RALM Transformer 層（無 RoPE）。
pub struct RalmLayer {
    input_layernorm: RMSNorm,
    self_attn: GQAAttention, // 無 RoPE 的 attention
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
}

impl RalmLayer {
    pub fn load(
        vb: &VarBuilder,
        i: usize,
        cfg: &LmConfig,
        use_kv_cache: bool,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
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

    /// 使用無 RoPE 的 attention。
    pub fn forward(&mut self, x: &Tensor, _step: usize) -> Result<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        // 實際上呼叫 forward_no_rope（跳過 RoPE）
        let h = self.forward_no_rope(&h)?;
        let h = (residual + h)?;

        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = (residual + h)?;
        Ok(h)
    }

    /// Attention forward without RoPE.
    /// Uses the standard attention but with a simple identity-style RoPE
    /// (cos=1, sin=0) so no positional information is added.
    fn forward_no_rope(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = x.shape().dims3()?;

        // Manual attention without RoPE:
        // q/k/v projections
        let q = self.self_attn.q_proj.forward(x)?; // [b, seq_len, num_heads * head_dim]
        let k = self.self_attn.k_proj.forward(x)?;
        let v = self.self_attn.v_proj.forward(x)?;

        let head_dim = self.self_attn.head_dim();
        let num_heads = self.self_attn.num_heads();
        let num_kv_heads = self.self_attn.num_kv_heads();

        let q = q.reshape((b, seq_len, num_heads, head_dim))?;
        let k = k.reshape((b, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((b, seq_len, num_kv_heads, head_dim))?;

        // GQA expand
        let group_size = num_heads / num_kv_heads;
        let k = k
            .unsqueeze(3)?
            .expand((b, seq_len, num_kv_heads, group_size, head_dim))?;
        let k = k.reshape((b, seq_len, num_heads, head_dim))?;
        let v = v
            .unsqueeze(3)?
            .expand((b, seq_len, num_kv_heads, group_size, head_dim))?;
        let v = v.reshape((b, seq_len, num_heads, head_dim))?;

        // Scaled dot-product attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let scale = (head_dim as f64).sqrt().recip();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b, seq_len, num_heads * head_dim))?;
        self.self_attn.o_proj.forward(&attn_output)
    }
}

/// RALM 完整模型（8 層）。
pub struct RALM {
    layers: Vec<RalmLayer>,
    norm: RMSNorm,
}

impl RALM {
    /// `num_layers` = `config.residual_lm_num_layers` (from VoxConfig, typically 8).
    pub fn load(
        vb: &VarBuilder, cfg: &LmConfig, num_layers: usize,
        dev: &Device, use_kv_cache: bool,
    ) -> Result<Self> {
        let norm = RMSNorm::load(vb, cfg.hidden_size, cfg.rms_norm_eps, "norm")?;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(RalmLayer::load(
                vb,
                i,
                cfg,
                use_kv_cache,
                cfg.max_position_embeddings,
                dev,
            )?);
        }
        Ok(Self { layers, norm })
    }

    pub fn forward(&mut self, x: &Tensor, step: usize) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, step)?;
        }
        self.norm.forward(&h)
    }
}
