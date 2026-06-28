//! UnifiedCFM — Flow Matching Euler solver wrapper around LocDiT。
//!
//! 對應 Python `UnifiedCFM` (from unified_cfm.py)。
//! 負責從 noise 開始，用 Euler ODE solver + CFG 逐步去噪，
//! 產生單個 patch（patch_size=4）的 acoustic features。

use super::loc_dit::LocDiT;
use candle_core::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

fn save_flat_tensor(t: &Tensor, prefix: &str) -> Result<()> {
    let flat = t.flatten_all()?.to_vec1::<f32>()?;
    let path = format!("{prefix}.f32");
    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, &bytes).map_err(|e| candle_core::Error::Msg(format!("save_flat: {e}")))
}

/// Flow Matching Euler solver 包裝 LocDiT estimator。
pub struct UnifiedCFM {
    pub estimator: LocDiT,
    pub inference_cfg_rate: f64,
    pub sigma_min: f64,
    pub solver: String,
    pub feat_dim: usize,
    pub target_dtype: DType,
    pub mean_mode: bool,
}

impl UnifiedCFM {
    pub fn new(estimator: LocDiT, cfg_rate: f64, sigma_min: f64, solver: &str, feat_dim: usize, mean_mode: bool) -> Self {
        let target_dtype = estimator.in_proj.weight().dtype();
        Self {
            estimator,
            inference_cfg_rate: cfg_rate,
            sigma_min,
            solver: solver.to_string(),
            feat_dim,
            target_dtype,
            mean_mode,
        }
    }

    /// Create a full tensor in the target dtype on the target device.
    /// On CUDA with non-F32 dtypes (BF16), creates on CPU first to avoid
    /// CUDA dtype conversion limitations in candle 0.9.2.
    fn make_full<S: Into<candle_core::Shape>>(&self, value: f32, shape: S, _dev: &Device) -> Result<Tensor> {
        let shape = shape.into();
        if self.target_dtype == DType::F32 {
            Tensor::full(value, &shape, _dev)
        } else {
            let cpu_t = Tensor::full(value, &shape, &Device::Cpu)?;
            let cvt = cpu_t.to_dtype(self.target_dtype)?;
            cvt.to_device(_dev)
        }
    }

    /// Create a randn tensor in the target dtype on the target device.
    fn make_randn<S: Into<candle_core::Shape>>(&self, shape: S, _dev: &Device) -> Result<Tensor> {
        let shape = shape.into();
        if self.target_dtype == DType::F32 {
            Tensor::randn(0.0f32, 1.0, &shape, _dev)
        } else {
            let cpu_t = Tensor::randn(0.0f32, 1.0, &shape, &Device::Cpu)?;
            let cvt = cpu_t.to_dtype(self.target_dtype)?;
            cvt.to_device(_dev)
        }
    }

    /// 產生單個 patch 的 acoustic features。
    ///
    /// 輸入：
    ///   mu:        [B, C] — 文字條件（dit_hidden，來自 LM 投影）
    ///   n_timesteps: usize — 擴散步數
    ///   patch_size: usize — patch 大小（通常為 4）
    ///   cond:      [B, C, T'] — 前一步預測特徵作為 prefix 條件
    ///   cfg_value: f64 — CFG guidance scale（1.0 = 無 CFG）
    ///   cancel:    Option<&AtomicBool> — 取消旗標
    ///
    /// 輸出：[B, C, patch_size] — 預測的 clean acoustic features
    pub fn forward(
        &mut self,
        mu: &Tensor,
        n_timesteps: usize,
        patch_size: usize,
        cond: &Tensor,
        cfg_value: f64,
        cancel: Option<&AtomicBool>,
    ) -> Result<Tensor> {
        let dev = mu.device();
        let _dtype = mu.dtype();
        let batch = mu.dim(0)?;
        let feat_dim = self.feat_dim; // C (fixed audio feature dimension, e.g. 64)

        // 初始噪聲 z ~ N(0, 1)，shape [B, C, patch_size]
        let z = self.make_randn(&[batch, feat_dim, patch_size], &dev)?;

        // 時間步長（均勻分布，從 1 → 0）
        // Python: t_span = linspace(1, 0, n_timesteps+1), then sway_sampling
        //   t_span = t_span + sway_coef * (cos(π/2 * t_span) - 1 + t_span)
        let t_span: Vec<f64> = (0..=n_timesteps)
            .map(|i| {
                let t = 1.0 - i as f64 / n_timesteps as f64;
                // sway_sampling_coef=1.0 (from Python default)
                t + 1.0 * ((std::f64::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
            })
            .collect();

        self.solve_euler(z, &t_span, mu, cond, cfg_value, cancel)
    }

    /// Euler ODE solver with CFG。
    ///
    /// Python 對應 `UnifiedCFM.solve_euler()`。
    fn solve_euler(
        &mut self,
        x: Tensor,
        t_span: &[f64],
        mu: &Tensor,
        cond: &Tensor,
        cfg_value: f64,
        cancel: Option<&AtomicBool>,
    ) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let feat_dim = x.dim(1)?;
        let x_len = x.dim(2)?;
        let _dtype = x.dtype();
        // Query device before moving x
        let dev = x.device().clone();

        let mut x_t = x;
        let num_steps = t_span.len() - 1;
        // Python: zero_init_steps = max(1, int(len(t_span) * 0.04))
        let zero_init_steps = std::cmp::max(1, (t_span.len() as f64 * 0.04) as usize);

        let mut t = t_span[0];

        for step in 0..num_steps {
            // 取消檢查
            if let Some(flag) = cancel {
                if flag.load(Ordering::SeqCst) {
                    return Err(candle_core::Error::Msg(
                        "Synthesis cancelled by user".into(),
                    ));
                }
            }

            let dt = t - t_span[step + 1];

            let dphi_dt = if step < zero_init_steps {
                // Python: zero velocity for first ~4% steps
                Tensor::zeros(&[batch, feat_dim, x_len], self.target_dtype, &dev)?
            } else {
                // CFG: double batch for conditional + unconditional
                // Python: mu_in is zeros, only first half gets actual mu (unconditional = null/zero conditioning)
                let x_2x = Tensor::cat(&[&x_t, &x_t], 0)?;
                let mu_null = self.make_full(0.0f32, &[batch, mu.dim(1)?], &dev)?;
                let mu_2x = Tensor::cat(&[mu, &mu_null], 0)?;
                let t_2x = self.make_full(t as f32, &[batch * 2], &dev)?;
                // Python: dt_in is zeros when mean_mode=false (standard inference)
                let dt_val = if self.mean_mode { dt as f32 } else { 0.0f32 };
                let dt_2x = self.make_full(dt_val, &[batch * 2], &dev)?;
                // Python: uncond path uses zeros_like(cond)
                let cond_null = self.make_full(0.0f32, &[batch, cond.dim(1)?, cond.dim(2)?], &dev)?;
                let cond_2x = Tensor::cat(&[cond, &cond_null], 0)?;

                let full_pred = self.estimator.forward(
                    &x_2x, &mu_2x, &t_2x, &cond_2x, &dt_2x,
                )?; // [2*B, C, T]

                // Debug: save velocity prediction from first non-zero step
                if step == zero_init_steps {
                    let dbg_pred = full_pred.to_dtype(DType::F32)?;
                    save_flat_tensor(&dbg_pred, "output/debug_cfm_velocity")?;
                    let dbg_mu = mu_2x.to_dtype(DType::F32)?;
                    save_flat_tensor(&dbg_mu, "output/debug_cfm_mu_2x")?;
                    let dbg_t = t_2x.to_dtype(DType::F32)?;
                    save_flat_tensor(&dbg_t, "output/debug_cfm_t")?;
                    let dbg_dt = dt_2x.to_dtype(DType::F32)?;
                    save_flat_tensor(&dbg_dt, "output/debug_cfm_dt")?;
                    let dbg_cond = cond_2x.to_dtype(DType::F32)?;
                    save_flat_tensor(&dbg_cond, "output/debug_cfm_cond")?;
                    let dbg_x_2x = x_2x.to_dtype(DType::F32)?;
                    save_flat_tensor(&dbg_x_2x, "output/debug_cfm_x_2x")?;
                }

                // Split: first half = conditional, second half = unconditional
                let pred_cond = full_pred.narrow(0, 0, batch)?;
                let pred_uncond = full_pred.narrow(0, batch, batch)?;

                // Python: use_cfg_zero_star=True with optimized_scale
                //   st_star = sum(pos * neg) / (sum(neg^2) + 1e-8)
                //   dphi_dt = st_star * neg + cfg * (pos - st_star * neg)
                // This adaptively scales the unconditional guidance direction
                // to align with the conditional prediction (zero-star CFG).
                let c_flat = pred_cond.reshape((batch, feat_dim * x_len))?;   // [B, C*T]
                let u_flat = pred_uncond.reshape((batch, feat_dim * x_len))?;
                let dot = (c_flat * &u_flat)?.sum_keepdim(1)?;                 // [B, 1]
                let eps = self.make_full(1e-8f32, &[1, 1], &dev)?;            // [1, 1] epsilon
                let norm_sq = (u_flat.sqr()?.sum_keepdim(1)? + &eps)?;         // [B, 1]
                let st_star = dot.broadcast_div(&norm_sq)?
                    .unsqueeze(2)?;                                            // [B, 1, 1] for [B, C, T] broadcast

                // dphi_dt = st_star * uncond + cfg * (cond - st_star * uncond)
                let neg_scaled = st_star.broadcast_mul(&pred_uncond)?;
                let diff = (pred_cond - &neg_scaled)?;
                (neg_scaled + (diff * cfg_value)?)?
            };

            // Euler step: x = x - dt * dphi_dt
            if dt.abs() > 1e-10 {
                let dt_t = self.make_full(dt as f32, &[batch, feat_dim, x_len], &dev)?;
                let step = dphi_dt.mul(&dt_t)?;
                x_t = x_t.sub(&step)?;
            }

            t = t_span[step + 1];
        }

        Ok(x_t)
    }
}
