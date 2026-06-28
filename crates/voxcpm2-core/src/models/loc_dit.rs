//! LocDiT — Local Diffusion Transformer (feat_decoder.estimator)。
//!
//! 對應 Python `VoxCPMLocDiT` (from local_dit_v2.py)。
//! 此模組是逐 patch 的 velocity estimator，由 `UnifiedCFM` wrapper 呼叫。
//!
//! Tensor prefix: `feat_decoder.estimator.*`

use super::{attention::GQAAttention, mlp::MLP, rms_norm::RMSNorm, rope::RoPE};
use crate::config::DitConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Save a tensor to a flat f32 file for Python analysis (same as pipeline save_debug_tensor)
fn save_tensor(t: &Tensor, prefix: &str) -> Result<()> {
    let flat = t.flatten_all()?.to_vec1::<f32>()?;
    let path = format!("{prefix}.f32");
    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, &bytes).map_err(|e| candle_core::Error::Msg(format!("save_tensor({prefix}): {e}")))
}

// ⚠ FlowMatchingScheduler will move to scheduler.rs after Z7 refactor
pub use self::scheduler::FlowMatchingScheduler;

mod scheduler {
    //! Flow Matching scheduler (temporary home — will be its own module).
    use candle_core::{Result, Tensor};
    use crate::config::CfmConfig;

    pub struct FlowMatchingScheduler {
        pub sigma_min: f64,
        pub solver: String,
        pub t_scheduler: String,
        pub inference_cfg_rate: f64,
    }

    impl FlowMatchingScheduler {
        pub fn from_config(cfg: &CfmConfig) -> Self {
            Self {
                sigma_min: cfg.sigma_min,
                solver: cfg.solver.clone(),
                t_scheduler: cfg.t_scheduler.clone(),
                inference_cfg_rate: cfg.inference_cfg_rate,
            }
        }

        pub fn timesteps(&self, num_steps: usize) -> Vec<f64> {
            match self.t_scheduler.as_str() {
                "log-norm" => {
                    let mut ts = Vec::with_capacity(num_steps + 1);
                    for i in 0..=num_steps {
                        let t = i as f64 / num_steps as f64;
                        let log_t = (t * 9.0 + 1.0).ln() / 10.0_f64.ln();
                        ts.push(log_t);
                    }
                    ts
                }
                _ => (0..=num_steps).map(|i| i as f64 / num_steps as f64).collect(),
            }
        }

        pub fn euler_step(&self, x: &Tensor, pred: &Tensor, dt: f64) -> Result<Tensor> {
            x + (pred * dt)?
        }

        pub fn apply_cfg(&self, cond: &Tensor, uncond: &Tensor) -> Tensor {
            if (self.inference_cfg_rate - 1.0).abs() < 1e-6 {
                return cond.clone();
            }
            let diff = (cond - uncond).unwrap();
            (uncond + (diff * self.inference_cfg_rate).unwrap()).expect("CFG apply failed")
        }
    }
}

/// 單一 DiT block（與 EncLayer 類似，但整合 time embedding）。
pub struct DiTBlock {
    norm1: RMSNorm,
    attn: GQAAttention,
    norm2: RMSNorm,
    mlp: MLP,
}

impl DiTBlock {
    pub fn load(vb: &VarBuilder, i: usize, cfg: &DitConfig, dev: &Device) -> Result<Self> {
        let kv_heads = cfg.num_heads / 8;
        let pp = vb.pp(format!("estimator.decoder.layers.{i}"));
        let eps = 1e-5;
        let norm1 = RMSNorm::load(&pp, cfg.hidden_dim, eps, "input_layernorm")?;
        let attn = GQAAttention::load(
            &pp,
            cfg.hidden_dim,
            cfg.num_heads,
            kv_heads,
            cfg.kv_channels,
            "self_attn",
            false,
            0,
            dev,
        )?;
        let norm2 = RMSNorm::load(&pp, cfg.hidden_dim, eps, "post_attention_layernorm")?;
        let mlp = MLP::load(&pp, cfg.hidden_dim, cfg.ffn_dim, "mlp")?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    pub fn forward(&mut self, x: &Tensor, rope: &RoPE, _step: usize) -> Result<Tensor> {
        let residual = x;
        let h = self.norm1.forward(x)?;
        let h = self.attn.forward(&h, rope, 0, false)?; // DiT uses bidirectional attention
        let h = (residual + h)?;
        let residual = h.clone();
        let h = self.norm2.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

/// LocDiT — velocity estimator，對應 Python `VoxCPMLocDiT.forward()`。
///
/// 輸入：
///   x:    [B, C, T]  — 噪聲潛變量（C=feat_dim=64, T=patch_size=4）
///   mu:   [B, C]     — 文字條件（dit_hidden，LM 投影）
///   t:    [B]        — 當前 diffusion timestep（f32 scalar）
///   cond: [B, C, T'] — 上一步預測特徵作為 prefix 條件
///   dt:   [B]        — 時間差（f32 scalar），目前未使用
///
/// 輸出：
///   [B, C, T] — velocity 預測
pub struct LocDiT {
    pub blocks: Vec<DiTBlock>,
    pub norm: RMSNorm,
    pub cond_proj: candle_nn::Linear,
    pub in_proj: candle_nn::Linear,
    pub out_proj: candle_nn::Linear,
    pub time_mlp_1: candle_nn::Linear,
    pub time_mlp_2: candle_nn::Linear,
    pub delta_time_mlp_1: candle_nn::Linear,
    pub delta_time_mlp_2: candle_nn::Linear,
    pub rope: RoPE,
    pub hidden_dim: usize,
    pub feat_dim: usize,
}

impl LocDiT {
    pub fn load(
        vb: &VarBuilder,
        cfg: &DitConfig,
        feat_dim: usize,
        rope_factors: Option<&[f64]>,
    ) -> Result<Self> {
        let dev = vb.device();
        let _ = DType::F32;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(DiTBlock::load(vb, i, cfg, dev)?);
        }
        let norm = RMSNorm::load(vb, cfg.hidden_dim, 1e-5, "estimator.decoder.norm")?;
        let cond_proj = candle_nn::linear(feat_dim, cfg.hidden_dim, vb.pp("estimator.cond_proj"))?;
        let in_proj = candle_nn::linear(feat_dim, cfg.hidden_dim, vb.pp("estimator.in_proj"))?;
        let out_proj = candle_nn::linear(cfg.hidden_dim, feat_dim, vb.pp("estimator.out_proj"))?;
        let time_mlp_1 = candle_nn::linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("estimator.time_mlp.linear_1"))?;
        let time_mlp_2 = candle_nn::linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("estimator.time_mlp.linear_2"))?;
        let delta_time_mlp_1 = candle_nn::linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("estimator.delta_time_mlp.linear_1"))?;
        let delta_time_mlp_2 = candle_nn::linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("estimator.delta_time_mlp.linear_2"))?;
        // Use short_factor from lm_config (Python: decoder_config = lm_config.model_copy(deep=True))
        // For sequences shorter than original_max_position_embeddings (which is 32768 and our DiT
        // sequences are at most ~40 tokens), short_factor is used.
        let rope = RoPE::new(8192, cfg.kv_channels, 10000.0, rope_factors, dev)?;

        Ok(Self {
            blocks,
            norm,
            cond_proj,
            in_proj,
            out_proj,
            time_mlp_1,
            time_mlp_2,
            delta_time_mlp_1,
            delta_time_mlp_2,
            rope,
            hidden_dim: cfg.hidden_dim,
            feat_dim,
        })
    }

    /// Velocity estimator forward，對應 Python `VoxCPMLocDiT.forward()`。
    ///
    /// 流程：
    ///   1. x [B, C, T] → transpose → in_proj → [B, T, hidden]
    ///   2. cond [B, C, T'] → transpose → cond_proj → [B, T', hidden]
    ///   3. time embedding: sin/cos(t) → time_mlp → [B, hidden]
    ///   4. delta time: sin/cos(dt) → delta_time_mlp → [B, hidden]
    ///   5. t_emb = time_mlp_out + delta_time_mlp_out
    ///   6. mu reshape: [B, C] → [B, -1, hidden] (view as 1 token)
    ///   7. concat: [mu(1), t_emb(1), cond(T'), x(T)] → decoder input
    ///   8. MiniCPM decoder (bidirectional, is_causal=false)
    ///   9. slice output: keep only x positions (remove mu + t + cond)
    ///   10. out_proj → transpose → [B, C, T]
    /// NOTE: uses `&mut self` because DiTBlock/GQAAttention may mutate cache.
    pub fn forward(
        &mut self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        let dev = x.device();
        // Use model weight dtype (BF16 on CUDA, F32 on CPU).
        // Convert all inputs to match weight dtype if needed.
        let model_dtype = self.in_proj.weight().dtype();
        let to_model = |t: &Tensor| -> Result<Tensor> {
            if t.dtype() != model_dtype { t.to_dtype(model_dtype) } else { Ok(t.clone()) }
        };
        let x_cvt = to_model(x)?;
        let mu_cvt = to_model(mu)?;
        let t_cvt = to_model(t)?;
        let cond_cvt = to_model(cond)?;
        let dt_cvt = to_model(dt)?;
        let batch = x_cvt.dim(0)?;

        // 1. in_proj: [B, C, T] → [B, T, C] → [B, T, hidden]
        let x_t = x_cvt.transpose(1, 2)?.contiguous()?;
        let h_x = self.in_proj.forward(&x_t)?; // [B, T, hidden]

        // 2. cond_proj: [B, C, T'] → [B, T', C] → [B, T', hidden]
        //    Handle zero-length cond (first patch, no prior features)
        let h_cond = if cond_cvt.dim(2)? == 0 {
            Tensor::zeros(&[batch, 0, self.hidden_dim], model_dtype, dev)?
        } else {
            let cond_t = cond_cvt.transpose(1, 2)?.contiguous()?;
            self.cond_proj.forward(&cond_t)?
        }; // [B, T', hidden]

        // 3–4. Time embedding: sin/cos → MLP
        let t_emb = self.time_embed_mlp(&t_cvt, dev, model_dtype)?;   // [B, hidden]
        let dt_emb = self.time_embed_mlp(&dt_cvt, dev, model_dtype)?; // [B, hidden]
        let t_combined = (t_emb + dt_emb)?;                 // [B, hidden]

        // 5. mu: [B, 2*hidden] → [B, 2, hidden] (Python: mu.view(B, -1, hidden))
        let h_mu = mu_cvt.reshape((batch, 2, self.hidden_dim))?; // [B, 2, hidden]

        // 6. t_combined: [B, hidden] → [B, 1, hidden]
        let h_t = t_combined.unsqueeze(1)?; // [B, 1, hidden]

        // 7. concat: [mu(2), t(1), cond(T'), x(T)]
        //    → [B, 2+1+T'+T, hidden]
        let decoder_in = Tensor::cat(&[&h_mu, &h_t, &h_cond, &h_x], 1)?;
        // Debug: save decoder input for analysis
        if decoder_in.dim(0)? > 1 {
            let dbg_dec_in = decoder_in.to_dtype(DType::F32)?;
            save_tensor(&dbg_dec_in, "output/debug_decoder_input")?;
        }

        // 8. Decoder forward (bidirectional, no causal mask)
        let mut h = decoder_in;
        for block in self.blocks.iter_mut() {
            h = block.forward(&h, &self.rope, 0)?;
        }
        h = self.norm.forward(&h)?;

        // 9. Slice output: keep only x positions (last T tokens)
        //    total = 2(mu) + 1(t) + T'(cond) + T(x)
        //    start_idx = 2 + 1 + T'
        let cond_len = cond_cvt.dim(2)?; // T'
        let x_len = x_cvt.dim(2)?;       // T
        let start_idx = 2 + 1 + cond_len;
        // Debug: save decoder output for cond vs uncond comparison
        if h.dim(0)? > 1 {
            let dbg_h = h.to_dtype(DType::F32)?;
            save_tensor(&dbg_h, "output/debug_decoder_output")?;
        }
        let h_x_out = h.narrow(1, start_idx, x_len)?; // [B, T, hidden]

        // 10. out_proj → transpose → [B, C, T]
        let out = self.out_proj.forward(&h_x_out)?; // [B, T, C]
        out.transpose(1, 2) // [B, C, T]
    }

    /// sin/cos time embedding + MLP (time_mlp).
    /// 對應 Python: `self.time_embeddings(t)` → `self.time_mlp(emb)`
    /// Python: SinusoidalPosEmb(hidden_dim) with scale=1000
    ///   half_dim = hidden_dim / 2
    ///   emb[i] = exp(-i * log(10000) / (half_dim - 1))
    ///   out = [sin(1000 * t * emb[i]), cos(1000 * t * emb[i])] for i in 0..half_dim
    fn time_embed_mlp(&self, t: &Tensor, dev: &Device, dtype: DType) -> Result<Tensor> {
        let half = self.hidden_dim / 2;
        let scale = 1000.0f64;
        // Convert to F32 first to support any input dtype (BF16 on CUDA)
        let t_f32 = t.to_dtype(DType::F32)?;
        let vals = t_f32.to_vec1::<f32>()?;
        let batch = vals.len();

        // Pre-compute frequency factors: exp(-i * log(10000) / (half-1))
        let log_10000 = (10000.0f64).ln();
        let half_minus_1 = (half - 1).max(1) as f64;
        let mut factors = Vec::with_capacity(half);
        for i in 0..half {
            factors.push((-(i as f64) * log_10000 / half_minus_1).exp());
        }

        // Python ordering: [sin(0..half), cos(0..half)]
        // For each batch element, compute sin and cos parts separately
        let mut sin_vals = Vec::with_capacity(batch * half);
        let mut cos_vals = Vec::with_capacity(batch * half);
        for &val in &vals {
            let v = scale * val as f64;
            for &freq in &factors {
                let angle = v * freq;
                sin_vals.push(angle.sin() as f32);
            }
            for &freq in &factors {
                let angle = v * freq;
                cos_vals.push(angle.cos() as f32);
            }
        }
        // Interleave sin then cos for each batch: [sin0..sin_half, cos0..cos_half]
        let mut emb = Vec::with_capacity(batch * self.hidden_dim);
        for b in 0..batch {
            let s_off = b * half;
            let c_off = b * half;
            for i in 0..half {
                emb.push(sin_vals[s_off + i]);
            }
            for i in 0..half {
                emb.push(cos_vals[c_off + i]);
            }
        }
        let emb_t = Tensor::from_vec(emb, &[batch, self.hidden_dim], dev)?.to_dtype(dtype)?;

        // time_mlp: Linear + SiLU + Linear
        let h = self.time_mlp_1.forward(&emb_t)?;
        let h = candle_nn::ops::silu(&h)?;
        self.time_mlp_2.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn scheduler_timesteps() {
        let sched = FlowMatchingScheduler {
            sigma_min: 1e-6,
            solver: "euler".into(),
            t_scheduler: "uniform".into(),
            inference_cfg_rate: 2.0,
        };
        let ts = sched.timesteps(10);
        assert_eq!(ts.len(), 11);
        assert!((ts[0] - 0.0).abs() < 1e-6);
        assert!((ts[10] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn locdit_forward_shape_test() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = DitConfig {
            hidden_dim: 256,
            ffn_dim: 1024,
            num_heads: 16,
            num_layers: 2,
            kv_channels: 64,
            mean_mode: false,
            latent_norm_scale: None,
            cfm_config: crate::config::CfmConfig {
                sigma_min: 1e-6,
                solver: "euler".into(),
                t_scheduler: "uniform".into(),
                inference_cfg_rate: 2.0,
                ..Default::default()
            },
        };
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let feat_dim = 64;
        let mut dit = LocDiT::load(&vb, &cfg, feat_dim, None)?;

        // 模擬 estimator forward call
        // mu = dit_hidden (2 * hidden_dim), in test hidden_dim=256 so mu_dim=512
        let batch = 1;
        let patch_size = 4;
        let cond_len = 4;
        let mu_dim = 2 * cfg.hidden_dim; // 512
        let x = Tensor::randn(0.0f32, 1.0, &[batch, feat_dim, patch_size], &dev)?;
        let mu = Tensor::randn(0.0f32, 1.0, &[batch, mu_dim], &dev)?;
        let t = Tensor::full(0.5f32, &[batch], &dev)?;
        let cond = Tensor::randn(0.0f32, 1.0, &[batch, feat_dim, cond_len], &dev)?;
        let dt = Tensor::full(0.1f32, &[batch], &dev)?;

        let out = dit.forward(&x, &mu, &t, &cond, &dt)?;
        assert_eq!(out.dims(), &[batch, feat_dim, patch_size],
            "LocDiT forward output shape");
        Ok(())
    }
}
