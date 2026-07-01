//! Autoregressive inference loop for VoxCPM2。
//!
//! 對應 Python `VoxCPM2Model._inference()`。
//!
//! 流程（Zero-shot mode）：
//!   1. TSLM base_lm full forward on text → kv_cache + lm_hidden [B, T, 2048]
//!   2. FSQ → fusion_concat_proj → RALM full forward → kv_cache + residual_hidden [B, T, 2048]
//!   3. Loop i = 0..max_len:
//!      a. lm_to_dit_proj(lm_hidden) + res_to_dit_proj(residual_hidden) → dit_hidden [B, 1, 2048]
//!      b. UnifiedCFM(mu=dit_hidden, patch_size=4, cond=prev_feat, n_timesteps, cfg)
//!         → pred_feat [B, 1, 4, 64]
//!      c. feat_encoder(音頻特徵) → enc_to_lm_proj → curr_embed [B, 1, 2048]
//!      d. stop_head check → break if stop
//!      e. TSLM.forward_step(curr_embed) → lm_hidden [B, 1, 2048]
//!      f. fsq_layer(lm_hidden) → lm_hidden_q
//!      g. fusion_concat_proj(concat(lm_hidden_q, curr_embed)) → [B, 1, 2048]
//!      h. RALM.forward_step(fusion) → residual_hidden [B, 1, 2048]
//!   4. 串接所有 pred_feat → [B, 64, total_frames]
//!   5. AudioVAE decode → waveform (在 pipeline.rs 中處理)

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

use crate::config::VoxConfig;
use crate::models::*;
use crate::pipeline::SynthRequest;

/// 從 2D tensor `[B, T, 2048]` 中取出最後一個位置的向量 `[B, 1, 2048]`。
fn last_hidden(h: &Tensor) -> candle_core::Result<Tensor> {
    let t = h.dim(1)?;
    h.narrow(1, t - 1, 1)
}

fn should_check_stop(step: usize, min_steps: usize) -> bool {
    // Python: `if i > min_len and stop_flag == 1`.
    step > min_steps
}

/// 完整自回歸生成迴圈。
///
/// # Args
/// * `main_vb` — model.safetensors 的 VarBuilder
/// * `config` — VoxCPM2 設定
/// * `req` — 合成請求參數
/// * `dev` — 裝置
/// * `input_ids` — 初始 text token IDs `[B, T]`
/// * `cancel` — 取消旗標
///
/// Voice Clone additional params:
/// * `init_combined_embeds` — Pre-computed combined embeddings `[B, T, 2048]`
///   for TSLM prefill. When Some, used instead of `embed_tokens(input_ids)`.
///   Already blended: `text_mask * embed(tokens) + (1-text_mask) * feat_embeds`.
/// * `init_feat_embeds` — Feat_embed part `[B, T, 2048]` for RALM prefill.
///   When None (text-only), zeros are used.
///
/// # Returns
/// * `[B, feat_dim, total_frames]` 生成的聲學特徵
pub fn generate_autoregressive(
    main_vb: &VarBuilder,
    config: &VoxConfig,
    req: &SynthRequest,
    dev: &Device,
    input_ids: &Tensor,
    cancel: Option<&AtomicBool>,
    max_len_base_tokens: usize,
) -> anyhow::Result<Tensor> {
    generate_autoregressive_with(
        main_vb,
        config,
        req,
        dev,
        input_ids,
        cancel,
        None,
        None,
        None,
        Some(max_len_base_tokens),
    )
}

/// 語音克隆專用：使用預先計算好的 combined embeddings + feat embeddings 進行自回歸生成。
pub fn generate_autoregressive_clone(
    main_vb: &VarBuilder,
    config: &VoxConfig,
    req: &SynthRequest,
    dev: &Device,
    input_ids: &Tensor,
    cancel: Option<&AtomicBool>,
    init_combined_embeds: &Tensor,
    init_feat_embeds: &Tensor,
    init_prev_feat: Option<&Tensor>,
    max_len_base_tokens: usize,
) -> anyhow::Result<Tensor> {
    generate_autoregressive_with(
        main_vb,
        config,
        req,
        dev,
        input_ids,
        cancel,
        Some(init_combined_embeds),
        Some(init_feat_embeds),
        init_prev_feat,
        Some(max_len_base_tokens),
    )
}

fn generate_autoregressive_with(
    main_vb: &VarBuilder,
    config: &VoxConfig,
    req: &SynthRequest,
    dev: &Device,
    input_ids: &Tensor,
    cancel: Option<&AtomicBool>,
    init_combined_embeds: Option<&Tensor>,
    init_feat_embeds: Option<&Tensor>,
    init_prev_feat: Option<&Tensor>,
    max_len_base_tokens: Option<usize>,
) -> anyhow::Result<Tensor> {
    let check_cancel = |name: &str| -> anyhow::Result<()> {
        if let Some(flag) = cancel {
            if flag.load(Ordering::Relaxed) {
                anyhow::bail!("Synthesis cancelled at: {name}");
            }
        }
        Ok(())
    };
    check_cancel("autoregressive-setup")?;

    let feat_dim = config.feat_dim; // 64
    let patch_size = config.patch_size; // 4

    // Dynamic max_len matching Python:
    //   max_len = min(text_tokens * retry_badcase_ratio_threshold + 10, global_max)
    // where retry_badcase_ratio_threshold = 6 (Python default), global_max = req.max_autoregressive_steps or 2000
    let global_max = req.max_autoregressive_steps.unwrap_or(2000);
    let seq_len = input_ids.dim(1)?;
    let max_len_base = max_len_base_tokens.unwrap_or(seq_len);
    let max_len = std::cmp::min(max_len_base * 6 + 10, global_max);
    eprintln!(
        "  [autoregressive] seq_len={seq_len} max_len_base={max_len_base} max_len={max_len} (global_max={global_max})"
    );

    // ── 載入 KV cache 版本的 TSLM ──
    let mut tslm_kv = TSLM::load(&main_vb.pp("base_lm"), &config.lm_config, dev, true)?;
    // Determine model dtype (BF16 on CUDA, F32 on CPU)
    let model_dtype = main_vb.dtype();
    // Voice clone: use pre-computed combined embeddings if provided.
    // Must match model dtype to avoid F32×BF16 matmul errors.
    let h_lm_init = if let Some(embeds) = init_combined_embeds {
        let embeds = if embeds.dtype() != model_dtype {
            embeds.to_dtype(model_dtype)?
        } else {
            embeds.clone()
        };
        tslm_kv.forward_embeds(&embeds, 0)?
    } else {
        tslm_kv.forward(input_ids, 0)?
    }; // [B, T, 2048], 同時填充 KV cache
    eprintln!(
        "  [autoregressive] TSLM init: seq_len={seq_len} h_lm_init.shape={:?} peak={:.6}",
        h_lm_init.shape(),
        tensor_peak(&h_lm_init)?
    );
    // Save TSLM output for comparison (feature: debug-tensors)
    #[cfg(feature = "debug-tensors")]
    save_debug_tensor(&h_lm_init, "debug_tslm_init")?;
    check_cancel("autoregressive-tslm-init")?;

    // ── 載入 KV cache 版本的 RALM ──
    let ralm_num_layers = config.residual_lm_num_layers;
    let mut ralm_kv = RALM::load(
        &main_vb.pp("residual_lm"),
        &config.lm_config,
        ralm_num_layers,
        dev,
        true,
    )?;

    // ── 載入其他組件 ──
    let enc_rope_factors: Option<Vec<f64>> = config
        .lm_config
        .rope_scaling
        .as_ref()
        .map(|rs| rs.short_factor.clone());
    let mut feat_enc = LocEnc::load(
        &main_vb,
        &config.encoder_config,
        feat_dim,
        patch_size,
        enc_rope_factors.as_deref(),
    )?;
    let fsq = FsqLayer::load(main_vb)?;
    let stop = StopHead::load(main_vb)?;
    let dit_rope_factors: Option<Vec<f64>> = config
        .lm_config
        .rope_scaling
        .as_ref()
        .map(|rs| rs.short_factor.clone());
    let dit = LocDiT::load(
        &main_vb.pp("feat_decoder"),
        &config.dit_config,
        feat_dim,
        dit_rope_factors.as_deref(),
    )?;
    let cfg_rate = config.dit_config.cfm_config.inference_cfg_rate;
    let sigma_min = config.dit_config.cfm_config.sigma_min;
    let solver = &config.dit_config.cfm_config.solver;
    let mean_mode = config.dit_config.mean_mode;
    let t_scheduler = &config.dit_config.cfm_config.t_scheduler;
    let t_scheduler_mean = config.dit_config.cfm_config.t_scheduler_mean;
    let t_scheduler_std = config.dit_config.cfm_config.t_scheduler_std;
    let mut cfm = UnifiedCFM::new(
        dit,
        cfg_rate,
        sigma_min,
        solver,
        feat_dim,
        mean_mode,
        t_scheduler,
        t_scheduler_mean,
        t_scheduler_std,
    );
    let lm_to_dit = candle_nn::linear(
        2048,
        config.dit_config.hidden_dim,
        main_vb.pp("lm_to_dit_proj"),
    )?;
    let res_to_dit = candle_nn::linear(
        2048,
        config.dit_config.hidden_dim,
        main_vb.pp("res_to_dit_proj"),
    )?;
    let fusion_concat_proj =
        candle_nn::linear_no_bias(4096, 2048, main_vb.pp("fusion_concat_proj"))?;

    // Python zero-shot prefill:
    //   residual_enc_inputs = fusion_concat_proj(cat((enc_outputs, feat_mask * feat_embed), dim=-1))
    // Zero-shot: feat_mask is false for all positions → second half is zeros.
    // Voice clone: feat_embed is non-zero for ref audio positions.
    let feat_part = match init_feat_embeds {
        Some(f) => {
            if f.dtype() != model_dtype {
                f.to_dtype(model_dtype)?
            } else {
                f.clone()
            }
        }
        None => Tensor::zeros(
            &[h_lm_init.dim(0)?, h_lm_init.dim(1)?, h_lm_init.dim(2)?],
            h_lm_init.dtype(),
            dev,
        )?,
    };
    let residual_prefill = Tensor::cat(&[&h_lm_init, &feat_part], 2)?;
    let residual_prefill = fusion_concat_proj.forward(&residual_prefill)?;

    // ── 自回歸迴圈 ──
    let mut all_feats: Vec<Tensor> = Vec::new();
    let mut prev_feat: Option<Tensor> = match init_prev_feat {
        Some(feat) if feat.dtype() != cfm.target_dtype => Some(feat.to_dtype(cfm.target_dtype)?),
        Some(feat) => Some(feat.clone()),
        None => None,
    };
    let min_steps = 2;
    let n_timesteps = req.inference_timesteps;
    let cfg_value = req.cfg_value;

    // h_lm 和 h_res 追蹤最後一個位置的 hidden state
    let mut h_lm = last_hidden(&h_lm_init)?; // [B, 1, 2048]
    let h_res_init = ralm_kv.forward(&residual_prefill, 0)?; // fills RALM KV cache with Python-matching prefill
    eprintln!(
        "  [autoregressive] RALM init: peak={:.6}",
        tensor_peak(&h_res_init)?
    );
    #[cfg(feature = "debug-tensors")]
    save_debug_tensor(&h_res_init, "debug_ralm_init")?;
    check_cancel("autoregressive-ralm-init")?;
    let mut h_res = last_hidden(&h_res_init)?; // [B, 1, 2048]

    for step in 0..max_len {
        if step % 10 == 0 {
            eprintln!("  [autoregressive] step {step}/{max_len}");
        }
        check_cancel(&format!("ar-step-{step}"))?;

        // a. lm_to_dit_proj(lm_hidden) + res_to_dit_proj(residual_hidden) → dit_hidden [B, 2048]
        //    lm_to_dit: [1024, 2048] 投影 2048→1024;  concat 兩個 → [B, 1, 2048] → squeeze → [B, 2048]
        let mu_input = Tensor::cat(&[lm_to_dit.forward(&h_lm)?, res_to_dit.forward(&h_res)?], 2)?; // [B, 1, 2048]
        let mu_input = mu_input.squeeze(1)?; // [B, 2048]

        // b. UnifiedCFM → pred_feat [B, feat_dim, patch_size]
        // Use the model's target dtype (BF16 on CUDA, F32 on CPU) for the initial cond.
        // Python: prefix_feat_cond starts as feat[:, -1, ...] which is [B, P, D]
        // transposed to [B, D, P] for CFM. For zero-shot, this is zeros([B, 64, 4]).
        let cond_dtype = cfm.target_dtype;
        let cond = prev_feat.clone().unwrap_or_else(|| {
            // Python uses [B, C=64, patch_size=4] of zeros (not zero-length!)
            Tensor::zeros(&[1, feat_dim, patch_size], cond_dtype, dev).unwrap()
        });
        // 每個 patch 用 req.seed + step 產生唯一種子，確保完全可重現
        let cfm_seed = req.seed.map(|base| base.wrapping_add(step as u64));
        let pred_feat = cfm.forward(
            &mu_input,
            n_timesteps,
            patch_size,
            &cond,
            cfg_value as f64,
            cancel,
            cfm_seed,
        )?;
        // pred_feat: [B, feat_dim, patch_size] = [1, 64, 4]
        // Diagnostic: log latent stats at each step
        let pf_f32 = pred_feat.to_dtype(DType::F32)?;
        let pf_peak = tensor_peak(&pf_f32)?;
        let pf_abs = pf_f32.abs()?.mean_all()?.to_vec0::<f32>()?;
        if step < 5 || step % 10 == 0 {
            eprintln!("  [AR diag] step {step}: pred_feat peak={pf_peak:.6} mean_abs={pf_abs:.6}");
        }
        all_feats.push(pred_feat.clone());

        // c. feat_encoder(音頻特徵) → curr_embed [B, 1, 2048]
        let curr_embed = feat_enc.encode(&pred_feat)?;

        // d. stop_head check
        if should_check_stop(step, min_steps) {
            let logits = stop.forward(&h_lm)?;
            if stop.should_stop(&logits)? {
                eprintln!("  [autoregressive] stop_head triggered at step {step}");
                break;
            }
        }

        // e. TSLM.forward_step(curr_embed) → lm_hidden [B, 1, 2048]
        let lm_out = tslm_kv.forward_step(&curr_embed, seq_len + step)?;

        // f. fsq_layer → lm_hidden_q
        // Python assigns this back to `lm_hidden`, so the next DiT step and stop head
        // both consume the quantized hidden state.
        h_lm = fsq.forward(&lm_out)?;

        // g. fusion_concat_proj(concat(lm_hidden_q, curr_embed)) → [B, 1, 2048]
        let fusion = Tensor::cat(&[&h_lm, &curr_embed], 2)?; // [B, 1, 4096]
        let fusion = fusion_concat_proj.forward(&fusion)?; // [B, 1, 2048]

        // h. RALM.forward_step(fusion) → residual_hidden [B, 1, 2048]
        h_res = ralm_kv.forward_step(&fusion, seq_len + step)?;

        prev_feat = Some(pred_feat);
    }

    check_cancel("autoregressive-done")?;

    if all_feats.is_empty() {
        anyhow::bail!("autoregressive loop produced no features");
    }

    // 串接所有 patch → [B, feat_dim, total_frames]
    let total_frames: usize = all_feats.iter().map(|t| t.dim(2).unwrap()).sum();
    eprintln!(
        "  [autoregressive] generated {total_frames} frames across {} patches",
        all_feats.len()
    );

    let cat_inputs: Vec<&Tensor> = all_feats.iter().collect();
    let latent = Tensor::cat(&cat_inputs, 2)?; // [B, 64, total_frames]
    Ok(latent)
}

fn tensor_peak(t: &Tensor) -> candle_core::Result<f32> {
    Ok(t.flatten_all()?
        .to_dtype(DType::F32)?
        .abs()?
        .max_all()?
        .to_vec0::<f32>()?)
}

#[cfg(feature = "debug-tensors")]
fn save_debug_tensor(t: &Tensor, name: &str) -> candle_core::Result<()> {
    let path = format!("output/{name}.f32");
    let flat = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, &bytes)
        .map_err(|e| candle_core::Error::Msg(format!("save_debug_tensor({name}): {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn stop_gate_matches_python_min_len_rule() {
        let min_steps = 2;
        assert!(!should_check_stop(0, min_steps));
        assert!(!should_check_stop(1, min_steps));
        assert!(!should_check_stop(2, min_steps));
        assert!(should_check_stop(3, min_steps));
    }

    #[test]
    fn last_hidden_extracts_final_position() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        // [B, T, C] = [1, 5, 2048]
        let h = Tensor::randn(0.0f32, 1.0, &[1, 5, 2048], &dev)?;
        let last = last_hidden(&h)?;
        // Should be [1, 1, 2048]
        assert_eq!(
            last.dims(),
            &[1, 1, 2048],
            "last_hidden should extract last position"
        );
        // Verify value matches h[:, -1:, :]
        let h_last = h.narrow(1, 4, 1)?;
        let diff = (last - h_last)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "last_hidden value mismatch");
        Ok(())
    }
}
