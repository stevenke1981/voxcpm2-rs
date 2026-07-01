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
    /// `causal`: apply causal mask (true for initial forward, false for auto-regressive steps)
    pub fn forward(&mut self, x: &Tensor, _step: usize, causal: bool) -> Result<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        // 實際上呼叫 forward_no_rope（跳過 RoPE）
        let h = self.forward_no_rope(&h, causal)?;
        let h = (residual + h)?;

        let residual = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = (residual + h)?;
        Ok(h)
    }

    /// Attention forward without RoPE (with KV cache support).
    ///
    /// Uses manual attention with KV cache so autoregressive steps can
    /// attend to all previous tokens even when given only the current token.
    /// `causal`: apply causal mask for initial full-sequence forward.
    fn forward_no_rope(&mut self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let (b, seq_len, _) = x.shape().dims3()?;

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

        // KV cache: concat new k/v with cached (No RoPE applied)
        let (k, v) = if let Some(cache) = &mut self.self_attn.kv_cache {
            // Use the KVCache's append method
            // Need to flatten k/v first since KVCache works at [b, seq, dim] level
            let k_flat = k.reshape((b, seq_len, num_kv_heads * head_dim))?;
            let v_flat = v.reshape((b, seq_len, num_kv_heads * head_dim))?;
            let (k_new, v_new) = cache.append(&k_flat, &v_flat, 0)?;
            // Reshape back
            let k_new = k_new.reshape((b, k_new.dim(1)?, num_kv_heads, head_dim))?;
            let v_new = v_new.reshape((b, v_new.dim(1)?, num_kv_heads, head_dim))?;
            (k_new, v_new)
        } else {
            (k, v)
        };
        let full_seq_len = k.dim(1)?;

        // GQA expand
        let group_size = num_heads / num_kv_heads;
        let k = k
            .unsqueeze(3)?
            .expand((b, full_seq_len, num_kv_heads, group_size, head_dim))?;
        let k = k.reshape((b, full_seq_len, num_heads, head_dim))?;
        let v = v
            .unsqueeze(3)?
            .expand((b, full_seq_len, num_kv_heads, group_size, head_dim))?;
        let v = v.reshape((b, full_seq_len, num_heads, head_dim))?;

        // Scaled dot-product attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = (head_dim as f64).sqrt().recip();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        // Apply causal mask if needed
        if causal && seq_len > 1 && seq_len == full_seq_len {
            let n = seq_len;
            let mut mask_vals = vec![0.0f32; n * n];
            for i in 0..n {
                for j in (i + 1)..n {
                    mask_vals[i * n + j] = -1e10;
                }
            }
            let mask = Tensor::from_slice(&mask_vals, &[n, n], x.device())?;
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
            let mask = mask.to_dtype(attn_weights.dtype())?;
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, num_heads * head_dim))?
            .contiguous()?;
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
        vb: &VarBuilder,
        cfg: &LmConfig,
        num_layers: usize,
        dev: &Device,
        use_kv_cache: bool,
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
        let causal = true; // RALM initial forward uses causal attention
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, step, causal)?;
        }
        self.norm.forward(&h)
    }

    /// Single-token forward step with pre-computed embedding.
    ///
    /// 對應自回歸迴圈中 `RALM(inputs_embeds=fusion, ...)`：
    /// - 輸入 embedding `[B, 1, hidden_size]`（fusion_concat_proj 輸出）
    /// - 透過所有 8 層 RALM（無 RoPE，使用 KV cache）
    /// - 回傳 `[B, 1, hidden_size]`
    pub fn forward_step(&mut self, input_embeds: &Tensor, step: usize) -> Result<Tensor> {
        let mut h = input_embeds.clone();
        let causal = false; // forward_step only has 1 token; KV cache handles causality
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, step, causal)?;
        }
        self.norm.forward(&h)
    }
}
