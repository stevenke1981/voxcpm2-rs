//! LocDiT — Local Diffusion Transformer + Flow Matching scheduler。
//!
//! 對應 tensor name prefix: 無獨立 prefix（DiT 整合在 feat_decoder.estimator 內）
//! 這裡實作 DiT block 與 CFG guidance + Euler ODE solver。

use super::{attention::GQAAttention, mlp::MLP, rms_norm::RMSNorm, rope::RoPE};
use crate::config::DitConfig;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// 單一 DiT block（與 EncLayer 類似，但整合 time embedding）。
pub struct DiTBlock {
    norm1: RMSNorm,
    attn: GQAAttention,
    norm2: RMSNorm,
    mlp: MLP,
}

impl DiTBlock {
    pub fn load(vb: &VarBuilder, i: usize, cfg: &DitConfig, dev: &Device) -> Result<Self> {
        // DiT 權重儲存在 feat_decoder.estimator.decoder.layers.* 中
        // 與 LocEnc decoder 共享，所以這裡只是個包裝
        let pp = vb.pp(format!("feat_decoder.estimator.decoder.layers.{i}"));
        let eps = 1e-5;
        let norm1 = RMSNorm::load(&pp, cfg.hidden_dim, eps, "input_layernorm")?;
        let attn = GQAAttention::load(
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
        let norm2 = RMSNorm::load(&pp, cfg.hidden_dim, eps, "post_attention_layernorm")?;
        let mlp = MLP::load(&pp, cfg.hidden_dim, cfg.ffn_dim, "mlp")?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    pub fn forward(&mut self, x: &Tensor, rope: &RoPE, step: usize) -> Result<Tensor> {
        let residual = x;
        let h = self.norm1.forward(x)?;
        let h = self.attn.forward(&h, rope, step)?;
        let h = (residual + h)?;
        let residual = h.clone();
        let h = self.norm2.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

/// Flow Matching scheduler。
pub struct FlowMatchingScheduler {
    pub sigma_min: f64,
    pub solver: String,
    pub t_scheduler: String,
    pub inference_cfg_rate: f64,
}

impl FlowMatchingScheduler {
    pub fn from_config(cfg: &crate::config::CfmConfig) -> Self {
        Self {
            sigma_min: cfg.sigma_min,
            solver: cfg.solver.clone(),
            t_scheduler: cfg.t_scheduler.clone(),
            inference_cfg_rate: cfg.inference_cfg_rate,
        }
    }

    /// 產生 ODE 時間步長。
    pub fn timesteps(&self, num_steps: usize) -> Vec<f64> {
        match self.t_scheduler.as_str() {
            "log-norm" => {
                // log-norm 調度
                let mut ts = Vec::with_capacity(num_steps + 1);
                for i in 0..=num_steps {
                    let t = i as f64 / num_steps as f64;
                    // log-norm: 在靠近 0 與 1 處集中取樣
                    let log_t = (t * 9.0 + 1.0).ln() / 10.0_f64.ln();
                    ts.push(log_t);
                }
                ts
            }
            _ => {
                // 均勻調度
                (0..=num_steps)
                    .map(|i| i as f64 / num_steps as f64)
                    .collect()
            }
        }
    }

    /// Euler 求解器：一步 ODE 更新。
    /// x: [batch, feat_dim, time] — 目前的噪聲潛變量
    /// pred: [batch, feat_dim, time] — 模型預測的 velocity
    /// dt: 步長
    pub fn euler_step(&self, x: &Tensor, pred: &Tensor, dt: f64) -> Result<Tensor> {
        // x + dt * pred
        x + (pred * dt)?
    }

    /// CFG (Classifier-Free Guidance) 合併條件與無條件預測。
    /// cond: 條件預測, uncond: 無條件預測, cfg_rate: guidance scale
    pub fn apply_cfg(&self, cond: &Tensor, uncond: &Tensor) -> Tensor {
        // pred = uncond + cfg_rate * (cond - uncond)
        // 若 inference_cfg_rate == 1.0，則等同 cond
        if (self.inference_cfg_rate - 1.0).abs() < 1e-6 {
            return cond.clone();
        }
        let diff = (cond - uncond).unwrap();
        (uncond + (diff * self.inference_cfg_rate).unwrap()).expect("CFG apply failed")
    }
}

/// LocDiT — 完整的 diffusion 生成器（包裝 feat_decoder.estimator）。
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
    pub scheduler: FlowMatchingScheduler,
    pub hidden_dim: usize,
    pub feat_dim: usize,
}

impl LocDiT {
    pub fn load(
        vb: &VarBuilder,
        cfg: &DitConfig,
        feat_dim: usize,
        scheduler: FlowMatchingScheduler,
    ) -> Result<Self> {
        let dev = vb.device();
        let _ = DType::F32;

        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(DiTBlock::load(vb, i, cfg, dev)?);
        }
        let norm = RMSNorm::load(
            vb,
            cfg.hidden_dim,
            1e-5,
            "feat_decoder.estimator.decoder.norm",
        )?;
        let cond_proj = candle_nn::linear(
            feat_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.cond_proj"),
        )?;
        let in_proj = candle_nn::linear(
            feat_dim,
            cfg.hidden_dim,
            vb.pp("feat_decoder.estimator.in_proj"),
        )?;
        let out_proj = candle_nn::linear(
            cfg.hidden_dim,
            feat_dim,
            vb.pp("feat_decoder.estimator.out_proj"),
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
        let rope = RoPE::new(8192, cfg.kv_channels, 10000.0, None, dev)?;

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
            scheduler,
            hidden_dim: cfg.hidden_dim,
            feat_dim,
        })
    }

    /// 產生 latent acoustic features。
    /// cond: [batch, time, hidden_dim] — 條件（來自 TSLM+RALM 的融合特徵）
    /// num_steps: 擴散步數
    /// seed: 隨機種子
    pub fn generate(
        &mut self,
        cond: &Tensor,
        num_steps: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let (batch, time, _cond_dim) = cond.shape().dims3()?;
        let dev = cond.device();

        // 初始噪聲 x_T ~ N(0, 1); Tensor::randn(mean, std, shape, device)
        let x = if let Some(s) = seed {
            // set_seed may fail on CPU backend; ignore gracefully
            let _ = dev.set_seed(s);
            Tensor::randn(0.0f32, 1.0, &[batch, self.feat_dim, time], dev)?
        } else {
            Tensor::randn(0.0f32, 1.0, &[batch, self.feat_dim, time], dev)?
        };

        let ts = self.scheduler.timesteps(num_steps);
        let mut x_t = x;

        for i in 0..num_steps {
            let t = ts[i];
            let dt = ts[i + 1] - ts[i];

            // Time embedding (簡單版: sin/cos 編碼)
            let t_embed = self.time_embed(t, dev)?; // [1, 1, hidden_dim]

            // 條件投影
            let cond_h = self.cond_proj.forward(cond)?;

            // 無條件預測 (用全零作為無條件)
            let zeros = Tensor::zeros(&[batch, time, self.hidden_dim], cond.dtype(), dev)?;

            // Velocity prediction 有條件
            let pred_cond = self.forward_velocity(&x_t, &cond_h, &t_embed, time)?;

            // 無條件 velocity
            let pred_uncond = self.forward_velocity(&x_t, &zeros, &t_embed, time)?;

            // CFG
            let pred = self.scheduler.apply_cfg(&pred_cond, &pred_uncond);

            // Euler step
            if dt.abs() > 1e-10 {
                x_t = self.scheduler.euler_step(&x_t, &pred, dt)?;
            }
        }

        // 最終輸出: [batch, feat_dim, time]
        Ok(x_t)
    }

    /// 計算 velocity（模型預測）。
    fn forward_velocity(
        &mut self,
        x: &Tensor,
        cond: &Tensor,
        t_embed: &Tensor,
        _time_steps: usize,
    ) -> Result<Tensor> {
        let x_t = x.transpose(1, 2)?; // [batch, time, feat_dim]
        let h = self.in_proj.forward(&x_t)?; // [batch, time, hidden_dim]
        let h = (h + cond)?;
        // 加上 time embedding (broadcast)
        let h = h.broadcast_add(t_embed)?;

        let mut h = h;
        for block in self.blocks.iter_mut() {
            h = block.forward(&h, &self.rope, 0)?;
        }

        h = self.norm.forward(&h)?;
        let out = self.out_proj.forward(&h)?; // [batch, time, feat_dim]
        out.transpose(1, 2) // [batch, feat_dim, time]
    }

    /// 簡單的 sin/cos time embedding。
    fn time_embed(&self, t: f64, dev: &Device) -> Result<Tensor> {
        let half = self.hidden_dim / 2;
        let mut emb = Vec::with_capacity(self.hidden_dim);
        for i in 0..half {
            let freq = 10000.0_f64.powf(-2.0 * i as f64 / self.hidden_dim as f64);
            let val = t * freq;
            emb.push(val.cos() as f32);
            emb.push(val.sin() as f32);
        }
        Tensor::from_vec(emb, self.hidden_dim, dev)?
            .unsqueeze(0)?
            .unsqueeze(0)
        // shape: [1, 1, hidden_dim]
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
    fn locdit_shape_test() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = DitConfig {
            hidden_dim: 256,
            ffn_dim: 1024,
            num_heads: 4,
            num_layers: 2,
            kv_channels: 64,
            mean_mode: false,
            cfm_config: crate::config::CfmConfig {
                sigma_min: 1e-6,
                solver: "euler".into(),
                t_scheduler: "uniform".into(),
                inference_cfg_rate: 2.0,
            },
        };
        let sched = FlowMatchingScheduler::from_config(&cfg.cfm_config);
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);

        let feat_dim = 64;
        let mut dit = LocDiT::load(&vb, &cfg, feat_dim, sched)?;
        let cond = Tensor::zeros(&[1, 4, feat_dim], DType::F32, &dev)?;
        let latent = dit.generate(&cond, 2, Some(42))?;
        assert_eq!(latent.dims(), &[1, 64, 4]);
        Ok(())
    }
}
