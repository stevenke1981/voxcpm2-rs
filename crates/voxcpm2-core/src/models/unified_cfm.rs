//! UnifiedCFM — Flow Matching Euler solver wrapper around LocDiT。
//!
//! 對應 Python `UnifiedCFM` (from unified_cfm.py)。
//! 負責從 noise 開始，用 Euler ODE solver + CFG 逐步去噪，
//! 產生單個 patch（patch_size=4）的 acoustic features。

use super::loc_dit::LocDiT;
use candle_core::{DType, Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};

/// Inverse error function (Abramowitz & Stegun approximation, max error ~1e-4).
/// Used for log-norm timestep computation.
fn erfinv(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x >= 1.0 {
        return sign * f64::INFINITY;
    }
    let a = 0.147;
    let ln1mx2 = (1.0 - x * x).ln();
    let part = 2.0 / (std::f64::consts::PI * a) + ln1mx2 / 2.0;
    let sqrt_part = (part * part - ln1mx2 / a).sqrt();
    sign * (sqrt_part - part).sqrt()
}

/// Compute log-norm timesteps: concentrates steps near t=0 (high-curvature region).
/// This is the default scheduler used in modern Flow Matching implementations.
fn lognorm_timesteps(n: usize, mean: f64, std: f64) -> Vec<f64> {
    let mut ts: Vec<f64> = (0..=n)
        .map(|i| {
            let u = i as f64 / n as f64; // 0.0 to 1.0
            let z = mean + std * 2.0_f64.sqrt() * erfinv(2.0 * u - 1.0);
            1.0 / (1.0 + (-z).exp()) // sigmoid
        })
        .collect();
    ts.reverse(); // 1 → 0
    ts
}

/// Compute uniform+sway timesteps (original VoxCPM2 default).
fn uniform_sway_timesteps(n: usize, sway_coef: f64) -> Vec<f64> {
    (0..=n)
        .map(|i| {
            let t = 1.0 - i as f64 / n as f64;
            t + sway_coef * ((std::f64::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
        })
        .collect()
}

#[cfg(feature = "debug-tensors")]
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
    pub t_scheduler: String,
    pub t_scheduler_mean: f64,
    pub t_scheduler_std: f64,
}

impl UnifiedCFM {
    pub fn new(
        estimator: LocDiT,
        cfg_rate: f64,
        sigma_min: f64,
        solver: &str,
        feat_dim: usize,
        mean_mode: bool,
        t_scheduler: &str,
        t_scheduler_mean: f64,
        t_scheduler_std: f64,
    ) -> Self {
        let target_dtype = estimator.in_proj.weight().dtype();
        Self {
            estimator,
            inference_cfg_rate: cfg_rate,
            sigma_min,
            solver: solver.to_string(),
            feat_dim,
            target_dtype,
            mean_mode,
            t_scheduler: t_scheduler.to_string(),
            t_scheduler_mean,
            t_scheduler_std,
        }
    }

    /// Create a full tensor in the target dtype on the target device.
    /// On CUDA with non-F32 dtypes (BF16), creates on CPU first to avoid
    /// CUDA dtype conversion limitations in candle 0.9.2.
    fn make_full<S: Into<candle_core::Shape>>(
        &self,
        value: f32,
        shape: S,
        _dev: &Device,
    ) -> Result<Tensor> {
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
    fn make_randn<S: Into<candle_core::Shape>>(
        &self,
        shape: S,
        seed: Option<u64>,
        _dev: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let cpu_t = match seed {
            Some(s) => {
                // Seeded: use StdRng + Box-Muller on CPU (always F32)
                let dims = shape.dims();
                let n: usize = dims.iter().product();
                let mut rng = StdRng::seed_from_u64(s);
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    let u1: f32 = rng.random::<f32>().clamp(f32::EPSILON, 1.0 - f32::EPSILON);
                    let u2: f32 = rng.random::<f32>().clamp(f32::EPSILON, 1.0 - f32::EPSILON);
                    let r = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f32::consts::PI * u2;
                    data.push(r * theta.cos());
                }
                Tensor::from_vec(data, &shape, &Device::Cpu)?
            }
            None => {
                // Unseeded: use candle's Tensor::randn
                Tensor::randn(0.0f32, 1.0, &shape, &Device::Cpu)?
            }
        };
        // Move to target device, converting dtype if needed
        if self.target_dtype == DType::F32 {
            cpu_t.to_device(_dev)
        } else {
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
    ///
    /// `seed` 控制初始噪聲的隨機種子，None 則使用非確定性隨機。
    pub fn forward(
        &mut self,
        mu: &Tensor,
        n_timesteps: usize,
        patch_size: usize,
        cond: &Tensor,
        cfg_value: f64,
        cancel: Option<&AtomicBool>,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let dev = mu.device();
        let _dtype = mu.dtype();
        let batch = mu.dim(0)?;
        let feat_dim = self.feat_dim; // C (fixed audio feature dimension, e.g. 64)

        // 初始噪聲 z ~ N(0, 1)，shape [B, C, patch_size]
        // 用 seed 確保可重現性
        let z = self.make_randn(&[batch, feat_dim, patch_size], seed, &dev)?;

        // 時間步長（從 1 → 0）
        // 支援兩種 scheduler：
        //   "log-norm" — 集中步數在 t≈0（高曲率區域），mean=-1.0, std=0.6 為預設
        //   "uniform" — 均勻分布 + sway_sampling（原始 VoxCPM2 行為）
        let t_span: Vec<f64> = if self.t_scheduler == "log-norm" {
            lognorm_timesteps(n_timesteps, self.t_scheduler_mean, self.t_scheduler_std)
        } else {
            // Default: uniform + sway sampling (coef=1.0 matches Python default)
            uniform_sway_timesteps(n_timesteps, 1.0)
        };

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
                if flag.load(Ordering::Relaxed) {
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
                // Python: uncond path also gets cond (both halves get same audio prefix)
                // cond_in[:b], cond_in[b:] = cond, cond — line 115 of unified_cfm.py
                let cond_2x = Tensor::cat(&[cond, cond], 0)?;

                let full_pred = self
                    .estimator
                    .forward(&x_2x, &mu_2x, &t_2x, &cond_2x, &dt_2x)?; // [2*B, C, T]

                // Debug: save velocity prediction from first non-zero step
                #[cfg(feature = "debug-tensors")]
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
                let c_flat = pred_cond.reshape((batch, feat_dim * x_len))?; // [B, C*T]
                let u_flat = pred_uncond.reshape((batch, feat_dim * x_len))?;
                let dot = (c_flat * &u_flat)?.sum_keepdim(1)?; // [B, 1]
                let eps = self.make_full(1e-8f32, &[1, 1], &dev)?; // [1, 1] epsilon
                let norm_sq = (u_flat.sqr()?.sum_keepdim(1)? + &eps)?; // [B, 1]
                let st_star = dot.broadcast_div(&norm_sq)?.unsqueeze(2)?; // [B, 1, 1] for [B, C, T] broadcast

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
