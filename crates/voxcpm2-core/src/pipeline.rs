use crate::{
    audio,
    autoregressive::{generate_autoregressive, generate_autoregressive_clone},
    config::VoxConfig,
    device::{self, DevicePreference},
    models::*,
    tokenizer::VoxTokenizer,
    weights,
};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthRequest {
    pub text: String,
    pub model_dir: Option<PathBuf>,
    pub output_path: PathBuf,
    pub device: String,
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub dry_run: bool,
    pub label_ai_generated: bool,
    pub seed: Option<u64>,
    pub voice_design: Option<String>,
    /// Optional post-gain factor. When set, the output waveform is multiplied
    /// by this factor. When None, a warning is emitted if peak is very low
    /// (which can happen with short text + few timesteps).
    pub post_gain: Option<f32>,
    /// Maximum autoregressive steps.
    /// Default formula: min(text_tokens * 6 + 10, max_autoregressive_steps).
    /// When None, uses 2000 as the hard upper bound (matching Python default).
    pub max_autoregressive_steps: Option<usize>,
    /// Timestep scheduler: "uniform" (default + sway) or "log-norm".
    /// Log-norm concentrates steps near t=0 for better quality.
    #[serde(default = "default_t_scheduler")]
    pub t_scheduler: String,
    /// Optional latent normalization scale before AudioVAE decode.
    /// Applied to CFM output: latent *= latent_norm_scale.
    /// Recommended: 0.7875 (= 1.26 / 1.60) to match Python latent std.
    #[serde(default)]
    pub latent_norm_scale: Option<f64>,
    /// Optional reference audio path for voice cloning.
    /// When set, the pipeline uses AudioVAE encoder to extract speaker
    /// characteristics and prefixes them in the autoregressive loop.
    #[serde(default)]
    pub ref_audio_path: Option<PathBuf>,
    /// Optional reference audio transcript (not yet used for alignment,
    /// but reserved for future phoneme-based conditioning).
    #[serde(default)]
    pub ref_transcript: Option<String>,
    /// Voice clone strength: 0.0 = no clone (text-only), 1.0 = full clone.
    /// Blends the reference audio feat_embeds with zeros.
    #[serde(default = "default_clone_strength")]
    pub clone_strength: f64,
    /// Optional JSON output path for machine-readable audio metrics.
    #[serde(default)]
    pub metrics_output_path: Option<PathBuf>,
}

fn default_clone_strength() -> f64 {
    1.0
}

fn default_t_scheduler() -> String {
    "uniform".into()
}

impl Default for SynthRequest {
    fn default() -> Self {
        Self {
            text: "Hello, world.".into(),
            model_dir: Some(PathBuf::from("models/VoxCPM2")),
            output_path: PathBuf::from("output/synth.wav"),
            device: "auto".into(),
            cfg_value: 2.5,
            inference_timesteps: 30,
            dry_run: false,
            label_ai_generated: true,
            seed: None,
            voice_design: None,
            post_gain: None,
            max_autoregressive_steps: None,
            t_scheduler: "uniform".into(),
            latent_norm_scale: None,
            ref_audio_path: None,
            ref_transcript: None,
            clone_strength: 1.0,
            metrics_output_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthResult {
    pub output_path: PathBuf,
    pub sample_rate: u32,
    pub samples: usize,
    pub device: String,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthMetricsReport {
    pub output_path: PathBuf,
    pub sample_rate: u32,
    pub samples: usize,
    pub device: String,
    pub dry_run: bool,
    pub is_clone: bool,
    pub label_ai_generated: bool,
    pub polish: Option<audio::AudioPolishReport>,
}

/// Cached model weights to avoid reloading from disk on every inference call.
/// Created lazily and invalidated when model_dir changes.
pub struct ModelCache {
    /// The model directory this cache was loaded from.
    pub model_dir: PathBuf,
    /// Main VarBuilder tensors (model.safetensors, ~4.6 GB on CUDA).
    pub main_tensors: Arc<HashMap<String, Tensor>>,
    /// Default dtype for the main model.
    pub main_default_dtype: DType,
    /// AudioVAE decoder tensors (audiovae.safetensors, fused weight_norm).
    pub audiovae_decoder_tensors: HashMap<String, Tensor>,
    /// Full AudioVAE tensors (encoder + decoder). Loaded on demand for clone.
    pub audiovae_all_tensors: Option<HashMap<String, Tensor>>,
    /// Tokenizer.
    pub tokenizer: VoxTokenizer,
    /// VoxConfig loaded from model_dir.
    pub config: VoxConfig,
}

impl ModelCache {
    /// Load (or reload) all model weights from `model_dir`.
    /// `need_encoder` — set to true when voice cloning is requested,
    /// triggering AudioVAE encoder tensor loading.
    pub fn load(model_dir: &Path, device: &Device, need_encoder: bool) -> anyhow::Result<Self> {
        // ── Main model tensors (model.safetensors) ──
        // Load once; convert BF16→F32 on CPU (same logic as weights::load_main_vb).
        let use_bf16 = matches!(device, Device::Cuda(_));
        let main_default_dtype = if use_bf16 { DType::BF16 } else { DType::F32 };
        let raw_tensors =
            candle_core::safetensors::load(&model_dir.join("model.safetensors"), device)?;
        let main_tensors: HashMap<String, Tensor> = if use_bf16 {
            raw_tensors
        } else {
            raw_tensors
                .into_iter()
                .map(|(k, t)| {
                    let t = if t.dtype() == DType::BF16 {
                        t.to_dtype(DType::F32).unwrap_or(t)
                    } else {
                        t
                    };
                    (k, t)
                })
                .collect()
        };
        let main_tensors = Arc::new(main_tensors);

        let audiovae_decoder_tensors = weights::load_audiovae_decoder_tensors(model_dir, device)?;
        let audiovae_all_tensors = if need_encoder {
            Some(weights::load_audiovae_all_tensors(model_dir, device)?)
        } else {
            None
        };
        let tokenizer = VoxTokenizer::from_model_dir(model_dir)?;
        let config = VoxConfig::load(model_dir)?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            main_tensors,
            main_default_dtype,
            audiovae_decoder_tensors,
            audiovae_all_tensors,
            tokenizer,
            config,
        })
    }

    /// Create a fresh VarBuilder from cached tensors for the main model.
    pub fn main_vb(&self, device: &Device) -> VarBuilder<'_> {
        let tensors: HashMap<String, Tensor> = (*self.main_tensors).clone();
        VarBuilder::from_tensors(tensors, self.main_default_dtype, device)
    }
}

pub struct VoxPipeline {
    pub device: Device,
    pub config: Option<VoxConfig>,
    /// Lazily cached model weights. Populated on first synthesize call.
    pub cache: Option<ModelCache>,
}

impl VoxPipeline {
    pub fn new(device_pref: &str, model_dir: Option<&Path>, dry_run: bool) -> anyhow::Result<Self> {
        let pref = DevicePreference::parse(device_pref)?;
        let device = device::select_device(pref)?;
        let config = if dry_run {
            None
        } else {
            // Fallback: use default model directory even when model_dir is not explicitly provided.
            let dir = model_dir.unwrap_or_else(|| Path::new("models/VoxCPM2"));
            Some(VoxConfig::load(dir)?)
        };
        Ok(Self {
            device,
            config,
            cache: None,
        })
    }

    /// Ensure model cache is populated for the given model_dir.
    /// Reloads only if model_dir changed from the cached one.
    pub fn ensure_cache(&mut self, model_dir: &Path, need_encoder: bool) -> anyhow::Result<()> {
        let should_reload = self.cache.as_ref().map_or(true, |c| {
            c.model_dir != model_dir || (need_encoder && c.audiovae_all_tensors.is_none())
        });
        if should_reload {
            eprintln!(
                "  [cache] loading model weights from {}...",
                model_dir.display()
            );
            let timer = std::time::Instant::now();
            self.cache = Some(ModelCache::load(model_dir, &self.device, need_encoder)?);
            // Also update self.config to match
            self.config = Some(self.cache.as_ref().unwrap().config.clone());
            eprintln!("  [cache] loaded in {:.1}s", timer.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Synthesize speech from text.
    ///
    /// `cancel` is an optional `AtomicBool` flag checked periodically during
    /// the long-running DiT diffusion loop.  When `true` the synthesis is
    /// aborted with a `Cancelled` error.
    pub fn synthesize(
        &mut self,
        req: &SynthRequest,
        cancel: Option<&AtomicBool>,
    ) -> anyhow::Result<SynthResult> {
        // Check cancel before starting
        if let Some(flag) = cancel {
            if flag.load(Ordering::Relaxed) {
                anyhow::bail!("Synthesis cancelled");
            }
        }

        let sample_rate = self
            .config
            .as_ref()
            .map(|c| c.audio_vae_config.out_sample_rate)
            .unwrap_or(48_000);

        let mut samples = if req.dry_run {
            audio::smoke_tone(&req.text, sample_rate)
        } else {
            let model_dir = req
                .model_dir
                .as_deref()
                .unwrap_or_else(|| Path::new("models/VoxCPM2"));
            // Ensure model weights are cached before synthesis
            let is_clone = req.ref_audio_path.is_some();
            self.ensure_cache(model_dir, is_clone)?;
            self.synthesize_real(req, model_dir, sample_rate, cancel)?
        };

        // Apply post-gain if specified, or warn if very quiet
        if let Some(gain) = req.post_gain {
            if (gain - 1.0).abs() > 1e-6 {
                for s in samples.iter_mut() {
                    *s *= gain;
                }
            }
        } else {
            let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            if peak < 1e-4 {
                tracing::warn!(
                    "Output peak very low ({:.6}). This is normal for short/noise latents. \
                     Set post_gain (e.g., 10.0) to amplify, or use more inference timesteps \
                     and longer text for natural volume.",
                    peak
                );
            }
        }

        let mut polish_report = None;
        if !req.dry_run {
            let polish = audio::polish_generated_speech(&mut samples, sample_rate);
            eprintln!(
                "  [audio] polish: dc={:.6} peak_before={:.6} peak_after={:.6} headroom_gain={:.6} quiet_rms={:.6}->{:.6} bg_gate={:.6} harsh_frames={}",
                polish.dc_offset,
                polish.peak_before,
                polish.peak_after,
                polish.headroom_gain,
                polish.quiet_rms_before,
                polish.quiet_rms_after,
                polish.background_gate_threshold,
                polish.harsh_frames_smoothed
            );
            polish_report = Some(polish);
        }

        audio::check_audio(&samples)?;
        audio::write_wav_f32(&req.output_path, &samples, sample_rate)?;
        let result = SynthResult {
            output_path: req.output_path.clone(),
            sample_rate,
            samples: samples.len(),
            device: device::device_label(&self.device),
            dry_run: req.dry_run,
        };

        if let Some(metrics_path) = &req.metrics_output_path {
            write_metrics_report(
                metrics_path,
                &SynthMetricsReport {
                    output_path: result.output_path.clone(),
                    sample_rate: result.sample_rate,
                    samples: result.samples,
                    device: result.device.clone(),
                    dry_run: result.dry_run,
                    is_clone: req.ref_audio_path.is_some(),
                    label_ai_generated: req.label_ai_generated,
                    polish: polish_report,
                },
            )?;
        }

        Ok(result)
    }

    /// Real VoxCPM2 inference pipeline:
    ///   text → tokenizer → TSLM → RALM → LocDiT → AudioVAE → waveform
    fn synthesize_real(
        &self,
        req: &SynthRequest,
        model_dir: &Path,
        _sample_rate: u32,
        cancel: Option<&AtomicBool>,
    ) -> anyhow::Result<Vec<f32>> {
        // Cancel check helpers
        let check_cancel = |name: &str| -> anyhow::Result<()> {
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    anyhow::bail!("Synthesis cancelled at: {name}");
                }
            }
            Ok(())
        };

        check_cancel("start")?;

        let dev = &self.device;
        eprintln!("  [pipe] using device: {:?}", dev);
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("VoxConfig required for real synthesis"))?;

        // ── Access cached weights ──
        let cache = self.cache.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Model cache not populated — call ensure_cache() before synthesize_real"
            )
        })?;
        let main_vb = cache.main_vb(dev);
        // AudioVAE — use CUDA device for AudioVAE if available
        let audiovae_dev = if matches!(dev, Device::Cuda(_)) {
            dev
        } else {
            &Device::Cpu
        };
        // Voice clone: use full audiovae tensors (encoder + decoder) from cache
        let is_clone = req.ref_audio_path.is_some();
        let audiovae_tensors = if is_clone {
            match &cache.audiovae_all_tensors {
                Some(tensors) => tensors.clone(),
                None => weights::load_audiovae_all_tensors(model_dir, audiovae_dev)?,
            }
        } else {
            cache.audiovae_decoder_tensors.clone()
        };

        // ── Tokenizer (cached) ──
        let tokenizer = cache.tokenizer.clone();
        let target_text = build_voice_design_text(&req.text, req.voice_design.as_deref());
        if looks_like_traditional_chinese_hint(&target_text) {
            eprintln!(
                "  [lang] Traditional Chinese text detected; VoxCPM2 may bias toward Cantonese. \
                 For Mandarin, use Simplified Chinese text."
            );
        }
        let tokens = tokenizer.encode_zero_shot(&target_text)?;
        if tokens.is_empty() {
            anyhow::bail!("tokenizer returned empty tokens");
        }

        // Voice clone: encode ref audio and build combined prefix
        // (needs TSLM for embed_text, so defer AR preparation until after TSLM is loaded)
        // We'll store the result here and use it during AR generation.
        let mut clone_prefix: Option<RefPrefixResult> = None;

        let input_ids = if let Some(ref_path) = &req.ref_audio_path {
            // Load TSLM first (needed for embed_text in encode_ref_prefix)
            let mut tslm_clone = TSLM::load(&main_vb.pp("base_lm"), &config.lm_config, dev, false)?;
            // Use cached audiovae encoder tensors if available
            let encoder_tensors = if is_clone {
                cache.audiovae_all_tensors.as_ref()
            } else {
                None
            };
            let prefix = encode_ref_prefix(
                model_dir,
                config,
                dev,
                ref_path,
                &tokenizer,
                &main_vb,
                encoder_tensors,
                &mut tslm_clone,
                &tokens,
                req.clone_strength,
            )?;
            let n_patches = prefix.n_patches;
            let combined_seq_len = prefix.combined_ids.dim(1)?;
            let combined_ids = prefix.combined_ids.clone();
            clone_prefix = Some(prefix);
            eprintln!(
                "  [clone] ready: {n_patches} ref patches, combined_seq_len={combined_seq_len}"
            );
            combined_ids
        } else {
            Tensor::from_slice(&tokens, &[1, tokens.len()], dev)?.to_dtype(DType::I64)?
        };

        let seq_len = input_ids.dim(1)?;
        let text_len = target_text.chars().count();
        if seq_len == 0 {
            anyhow::bail!("tokenizer returned empty tokens");
        }

        // ── TSLM forward ──
        check_cancel("TSLM")?;
        let mut tslm = TSLM::load(&main_vb.pp("base_lm"), &config.lm_config, dev, false)?;
        let h_tslm = tslm.forward(&input_ids, 0)?; // [1, seq_len, 2048]
        eprintln!(
            "  [pipe] TSLM: seq_len={seq_len} text_len={text_len} h_tslm.shape={:?}",
            h_tslm.shape()
        );
        #[cfg(feature = "debug-tensors")]
        save_debug_tensor(&h_tslm, "debug_pipe_tslm")?;
        let (tslm_mean, tslm_std, tslm_peak) = tensor_stats(&h_tslm)?;
        eprintln!("  [pipe] TSLM: seq_len={seq_len} text_len={text_len} mean={tslm_mean:.6} std={tslm_std:.6} peak={tslm_peak:.6}");

        // ── RALM forward ──
        check_cancel("RALM")?;
        let ralm_num_layers = config.residual_lm_num_layers;
        let mut ralm = RALM::load(
            &main_vb.pp("residual_lm"),
            &config.lm_config,
            ralm_num_layers,
            dev,
            false,
        )?;
        let h_ralm = ralm.forward(&h_tslm, 0)?; // [1, seq_len, 2048]
        #[cfg(feature = "debug-tensors")]
        save_debug_tensor(&h_ralm, "debug_pipe_ralm")?;
        let (ralm_mean, ralm_std, ralm_peak) = tensor_stats(&h_ralm)?;
        let h_tslm_peak = tensor_peak(&h_tslm)?;
        eprintln!("  [pipe] RALM: mean={ralm_mean:.6} std={ralm_std:.6} peak={ralm_peak:.6} (TSLM peak={h_tslm_peak:.6})");

        // ── Text → DiT condition (1024-dim) ──
        check_cancel("cond")?;
        let (lm_to_dit, res_to_dit) = weights::load_text_to_dit_projections(&main_vb)?;
        let cond_text = (lm_to_dit.forward(&h_tslm)? + res_to_dit.forward(&h_ralm)?)?; // [1, seq_len, 1024]
        #[cfg(feature = "debug-tensors")]
        save_debug_tensor(&cond_text, "debug_pipe_cond_text")?;
        let (ct_mean, ct_std, ct_peak) = tensor_stats(&cond_text)?;
        eprintln!(
            "  [pipe] cond_text (lm+res→1024): mean={ct_mean:.6} std={ct_std:.6} peak={ct_peak:.6}"
        );

        // ── Apply request-level scheduler override via cloned config ──
        let mut ar_config = config.clone();
        match req.t_scheduler.as_str() {
            "log-norm" | "uniform" => {
                ar_config.dit_config.cfm_config.t_scheduler = req.t_scheduler.clone();
            }
            _ => {
                tracing::warn!(
                    "Unknown t_scheduler '{}', using config default '{}'",
                    req.t_scheduler,
                    ar_config.dit_config.cfm_config.t_scheduler,
                );
            }
        }
        eprintln!(
            "  [pipe] CFM scheduler: {} (mean={:.1}, std={:.1})",
            ar_config.dit_config.cfm_config.t_scheduler,
            ar_config.dit_config.cfm_config.t_scheduler_mean,
            ar_config.dit_config.cfm_config.t_scheduler_std,
        );

        // ── Autoregressive generation ──
        check_cancel("autoregressive")?;
        let latent = if let Some(prefix) = &clone_prefix {
            generate_autoregressive_clone(
                &main_vb,
                &ar_config,
                req,
                dev,
                &prefix.combined_ids,
                cancel,
                &prefix.combined_embeds,
                &prefix.feat_embeds,
            )?
        } else {
            generate_autoregressive(&main_vb, &ar_config, req, dev, &input_ids, cancel)?
        };
        // latent shape: [1, feat_dim, seq_len]
        let latent_peak = latent
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_dtype(DType::F32)?
            .to_vec0::<f32>()?;
        if latent_peak < 0.001 {
            tracing::warn!("DiT latent very quiet (peak={:.6})", latent_peak);
        }

        // ── Optional latent normalization (match Python distribution) ──
        // Default: per-channel variance capping + global scale.
        // Strategy:
        //   1. Compute per-channel mean and std across time frames.
        //   2. Cap per-channel std to PYTHON_CH_STD_MAX (1.6):
        //      For channels where std > cap, normalize to N(0, cap).
        //      Narrow channels retain their natural profile.
        //   3. Apply global soft-scale: PYTHON_GLOBAL_STD / RUST_GLOBAL_STD.
        // This preserves narrow-channel variance structure (speech info)
        // while preventing wide-channel OOD saturation of AudioVAE.
        // Use --latent-norm <scale> for pure global scaling.
        // Use --latent-norm 1.0 to disable.
        const PYTHON_CH_STD_CAP: f64 = 1.6; // cap per-channel std to this
        const PYTHON_GLOBAL_STD: f64 = 1.26;
        const RUST_GLOBAL_STD: f64 = 1.60;
        const DEFAULT_GLOBAL_SCALE: f64 = PYTHON_GLOBAL_STD / RUST_GLOBAL_STD; // 0.7875

        fn is_disabled(s: f64) -> bool {
            (s - 1.0).abs() < 1e-6
        }

        let latent_for_vae = match req.latent_norm_scale {
            None => {
                // Default: per-channel variance cap + global scale
                let d = latent.dtype();
                let latent_work = latent.to_dtype(DType::F32)?;
                let (_b, c, _t) = latent_work.dims3()?;

                // 1. Per-channel mean and std
                let mean = latent_work.mean_keepdim(2)?; // [1, c, 1]
                let var = latent_work.var_keepdim(2)?; // [1, c, 1]
                let min_var = Tensor::full(1e-8f32, &[1, c, 1], dev)?;
                let std_ch = var.maximum(&min_var)?.sqrt()?; // [1, c, 1]

                // 2. Cap per-channel std: for channels where std > cap,
                //    scale them down so that max(std) = PYTHON_CH_STD_CAP.
                //    Scale factor per channel: min(1.0, cap / std)
                let cap_t = Tensor::full(PYTHON_CH_STD_CAP as f32, &[1, c, 1], dev)?;
                // scale_per_ch = min(1.0, cap / std)  => for each ch: if std > cap, std * (cap/std) = cap
                // inv_ratio = min(1.0, cap / std) = cap / max(std, cap)
                // We compute: safe_std = max(std, cap), then inv_ratio = cap / safe_std
                let std_safe = std_ch.maximum(&cap_t)?; // [1, c, 1]
                let inv_ratio = cap_t.broadcast_div(&std_safe)?; // [1, c, 1], in (0, 1]

                // Apply per-channel scale: center and rescale
                let centered = latent_work.broadcast_sub(&mean)?; // [1, c, t]
                let capped = centered.broadcast_mul(&inv_ratio)?; // [1, c, t]
                                                                  // Re-add mean (already near 0 for each channel)
                let re_centered = capped.broadcast_add(&mean)?; // [1, c, t]

                // 3. Global soft-scale toward Python distribution
                let scale_t = Tensor::full(DEFAULT_GLOBAL_SCALE as f32, &[1, 1, 1], dev)?;
                let scaled = re_centered.broadcast_mul(&scale_t)?; // [1, c, t]

                let out = scaled.to_dtype(d)?;

                // Diagnostics
                let ch_stds: Vec<f32> = std_ch.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?;
                let avg_std: f32 = ch_stds.iter().sum::<f32>() / ch_stds.len() as f32;
                let min_ch_std = ch_stds.iter().cloned().fold(f32::MAX, f32::min);
                let max_ch_std = ch_stds.iter().cloned().fold(f32::MIN, f32::max);
                let capped_chs = ch_stds
                    .iter()
                    .filter(|&&s| s > PYTHON_CH_STD_CAP as f32)
                    .count();
                eprintln!(
                    "  [pipe] latent_norm: var-cap({PYTHON_CH_STD_CAP})+scale({DEFAULT_GLOBAL_SCALE}) \
                     ch_stds: mean={avg_std:.4} range=[{min_ch_std:.4}, {max_ch_std:.4}] capped={capped_chs}/{c}"
                );
                out
            }
            Some(scale) if is_disabled(scale) => {
                eprintln!("  [pipe] latent_norm: disabled (scale=1.0)");
                latent.clone()
            }
            Some(scale) => {
                // Custom global scale (backward compat)
                let dtype = latent.dtype();
                let scale_t = Tensor::full(scale as f32, &[1, 1, 1], dev)?.to_dtype(dtype)?;
                let scaled = latent.broadcast_mul(&scale_t)?;
                eprintln!("  [pipe] latent_norm: global scale={scale}");
                scaled
            }
        };

        // ── AudioVAE decode ──
        let vae = crate::models::AudioVAE::load(&audiovae_tensors, &config.audio_vae_config)?;
        let latent_vae = latent_for_vae
            .to_device(audiovae_dev)?
            .to_dtype(DType::F32)?;
        eprintln!(
            "  [pipe] AudioVAE decode on device: {:?}",
            latent_vae.device()
        );
        {
            let latent_peak = tensor_peak(&latent_vae)?;
            let latent_stats = tensor_stats(&latent_vae)?;
            eprintln!(
                "  [pipe] AudioVAE latent: peak={:.6} mean={:.6} std={:.6}",
                latent_peak, latent_stats.0, latent_stats.1
            );
        }
        // Save latent for Python verification
        if let Ok(latent_vae_2d) = latent_vae.squeeze(0) {
            let latent_flat = latent_vae_2d.flatten_all()?.to_vec1::<f32>()?;
            let mut latent_bytes = Vec::with_capacity(latent_flat.len() * 4);
            for &v in &latent_flat {
                latent_bytes.extend_from_slice(&v.to_le_bytes());
            }
            if std::fs::write("output/latent_rust.f32", &latent_bytes).is_ok() {
                eprintln!(
                    "  [pipe] Saved latent to output/latent_rust.f32 ({} frames x 64 ch)",
                    latent_vae.dim(1).unwrap_or(0)
                );
            }
        }
        let waveform = vae.decode(&latent_vae)?;
        // waveform shape: [1, 1, samples]

        // Flatten to Vec<f32>
        let vals = waveform.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vals)
    }
}

/// Result of encoding a reference audio for voice cloning.
pub struct RefPrefixResult {
    /// Combined token IDs (ref + text): [1, T_total]
    pub combined_ids: Tensor,
    /// Combined embeddings (blended text_embed + feat_embed): [1, T_total, 2048]
    pub combined_embeds: Tensor,
    /// Feat_embed part (ref patches, zeros elsewhere): [1, T_total, 2048]
    pub feat_embeds: Tensor,
    /// Number of ref audio patches
    pub n_patches: usize,
}

/// Encode a reference audio file and build the combined prefix for voice cloning.
///
/// Steps:
///   1. Load reference audio WAV (any sample rate), resample to 16000 Hz
///   2. Encode through AudioVAE CausalEncoder → [1, 64, T']
///   3. Partition into patches of `patch_size` (=4) frames
///   4. For each patch, encode through LocEnc (feat_encoder) → [1, 1, 2048]
///   5. Build combined token IDs:
///      `[ref_audio_start, pad×n_patches, ref_audio_end, text_ids...]`
///   6. Build text_mask = 1 for token positions, 0 for audio (pad) positions
///   7. Build combined_embed = text_mask * embed(tokens) + (1-text_mask) * feat_embeds
///
/// # Args
/// * `model_dir` — model directory with `audiovae.safetensors` and `model.safetensors`
/// * `config` — VoxConfig
/// * `dev` — compute device
/// * `ref_audio_path` — path to reference WAV file
/// * `tokenizer` — VoxTokenizer (for special tokens)
/// * `main_vb` — model.safetensors VarBuilder (for feat_encoder weights)
/// * `tslm` — TSLM instance for embed_token lookups
/// * `text_ids` — the text-only token IDs (from `tokenizer.encode_zero_shot`)
/// * `clone_strength` — blend factor for reference audio features (0.0=no clone, 1.0=full)
pub fn encode_ref_prefix(
    model_dir: &Path,
    config: &VoxConfig,
    dev: &Device,
    ref_audio_path: &Path,
    tokenizer: &VoxTokenizer,
    main_vb: &VarBuilder,
    // Optional pre-loaded audoVAE encoder tensors; loaded on demand if None.
    audiovae_encoder_tensors: Option<&HashMap<String, Tensor>>,
    tslm: &mut TSLM,
    text_ids: &[u32],
    clone_strength: f64,
) -> anyhow::Result<RefPrefixResult> {
    // ── 1. Load and resample reference audio ──
    let (samples, src_rate) = audio::load_wav_mono(ref_audio_path)?;
    eprintln!(
        "  [clone] loaded ref audio: {} samples at {} Hz",
        samples.len(),
        src_rate
    );
    let end_time = std::time::Instant::now();

    let mut samples_16k = audio::resample(&samples, src_rate, 16000);
    let ref_polish = audio::polish_clone_reference_audio(&mut samples_16k, 16000);
    eprintln!(
        "  [clone] ref polish: dc={:.6} peak_before={:.6} peak_after={:.6} quiet_rms={:.6}->{:.6} bg_gate={:.6} harsh_frames={}",
        ref_polish.dc_offset,
        ref_polish.peak_before,
        ref_polish.peak_after,
        ref_polish.quiet_rms_before,
        ref_polish.quiet_rms_after,
        ref_polish.background_gate_threshold,
        ref_polish.harsh_frames_smoothed
    );
    // Auto-trim to MAX_REF_SECS to prevent CUDA OOM in AudioVAE encoder
    // (model.1 conv creates [1, 2048, T], ~23 GB for 178s audio).
    const MAX_REF_SECS: f64 = 30.0;
    let max_samples = (MAX_REF_SECS * 16000.0) as usize;
    if samples_16k.len() > max_samples {
        eprintln!(
            "  [clone] trimming {} samples ({:.2}s) → {max_samples} samples ({MAX_REF_SECS}s) to prevent OOM",
            samples_16k.len(),
            samples_16k.len() as f64 / 16000.0,
        );
        samples_16k.truncate(max_samples);
    }
    let audio_t =
        Tensor::from_slice(&samples_16k, &[1, 1, samples_16k.len()], dev)?.to_dtype(DType::F32)?;
    eprintln!(
        "  [clone] resampled to 16 kHz: {} samples ({:.2}s)",
        samples_16k.len(),
        samples_16k.len() as f64 / 16000.0
    );

    // ── 2. Load AudioVAE encoder tensors (use cached if provided) ──
    let audiovae_tensors = match audiovae_encoder_tensors {
        Some(full_tensors) => {
            // Filter to encoder.* only from the full set
            let enc: HashMap<String, Tensor> = full_tensors
                .iter()
                .filter(|(k, _)| k.starts_with("encoder."))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            enc
        }
        None => weights::load_audiovae_encoder_tensors(model_dir, dev)?,
    };
    let encoder = AudioVAE::load_encoder(&audiovae_tensors, &config.audio_vae_config)?;
    let latent = encoder.encode(&audio_t)?; // [1, 64, T']
    let (_b, _c, t_total) = latent.dims3()?;
    let patch_size = config.patch_size; // 4
    let n_patches = t_total / patch_size;
    if n_patches == 0 {
        anyhow::bail!(
            "Reference audio too short: {t_total} latent frames, need at least {patch_size} (>= {:.2}s)",
            patch_size as f64 * 640.0 / 16000.0
        );
    }
    // Safety cap: ~256 patches = ~41s ref audio at 16kHz.
    // Beyond this, the autoregressive prefix becomes too long for VRAM.
    const MAX_PATCHES: usize = 256;
    let n_patches = n_patches.min(MAX_PATCHES);
    eprintln!(
        "  [clone] AudioVAE encoder: latent [1, 64, {t_total}] → {n_patches} patches of {patch_size}",
    );

    // ── 3. Load feat_encoder (LocEnc) for patch encoding ──
    let enc_rope_factors: Option<Vec<f64>> = config
        .lm_config
        .rope_scaling
        .as_ref()
        .map(|rs| rs.short_factor.clone());
    let feat_dim = config.feat_dim;
    let mut feat_enc = LocEnc::load(
        main_vb,
        &config.encoder_config,
        feat_dim,
        patch_size,
        enc_rope_factors.as_deref(),
    )?;

    // ── 4. Encode each ref audio patch ──
    // LocEnc weights are BF16 (from model.safetensors on CUDA), but AudioVAE latent
    // is F32. Convert patch to model dtype (BF16) before encoding to match.
    let model_dtype = main_vb.dtype();
    let mut patch_embeds: Vec<Tensor> = Vec::with_capacity(n_patches);
    for i in 0..n_patches {
        let patch = latent.narrow(2, i * patch_size, patch_size)?; // [1, 64, 4]
        let patch = patch.to_dtype(model_dtype)?; // BF16 to match feat_enc weights
        let embed = feat_enc.encode(&patch)?; // [1, 1, 2048]
        patch_embeds.push(embed);
    }
    // ref_feat: [n_patches, 1, 2048] — cat along dim 0 (stack would create extra dim)
    let mut ref_feat = Tensor::cat(&patch_embeds, 0)?;
    // Convert back to F32 for mask/combine operations (text_mask is F32)
    ref_feat = ref_feat.to_dtype(DType::F32)?;
    // Apply clone_strength: blend between full clone (1.0) and no clone (0.0)
    if clone_strength != 1.0 {
        let scale = Tensor::full(clone_strength as f32, &[], dev)?;
        ref_feat = ref_feat.mul(&scale)?;
    }
    eprintln!(
        "  [clone] feat_encoder: {n_patches} patches → [{n_patches}, 1, 2048] strength={clone_strength}",
    );

    // ── 5. Build combined token IDs ──
    let special = tokenizer.special_tokens();
    let pad_id = special.unk_token; // use <unk> as pad token for ref audio positions

    let n_ref_tokens = 2 + n_patches; // ref_start + n_patches + ref_end
    let mut combined_ids = Vec::with_capacity(n_ref_tokens + text_ids.len());
    combined_ids.push(special.ref_audio_start);
    for _ in 0..n_patches {
        combined_ids.push(pad_id);
    }
    combined_ids.push(special.ref_audio_end);
    combined_ids.extend_from_slice(text_ids);
    let total_len = combined_ids.len();

    eprintln!(
        "  [clone] token sequence: ref_start={} pad×{n_patches} ref_end={} text={} total={total_len}",
        special.ref_audio_start,
        special.ref_audio_end,
        text_ids.len(),
    );

    // ── 6. Build text_mask and feat_embeds_full ──
    // Text positions (text_mask=1): ref_start (0), ref_end (n_patches+1),
    //   and all text positions (n_patches+2..total)
    // Audio positions (text_mask=0): the pad positions (1..n_patches+1)
    let one = Tensor::full(1.0f32, &[1, 1, 2048], dev)?;
    let zero = Tensor::zeros(&[1, 1, 2048], DType::F32, dev)?;

    let mut text_mask_parts: Vec<Tensor> = Vec::with_capacity(total_len);
    let mut feat_embed_parts: Vec<Tensor> = Vec::with_capacity(total_len);

    // ref_audio_start position: text, no feat
    text_mask_parts.push(one.clone()); // [1, 1, 2048]
    feat_embed_parts.push(zero.clone()); // [1, 1, 2048]

    // Pad positions (n_patches): audio, feat from encoder
    for i in 0..n_patches {
        text_mask_parts.push(zero.clone());
        feat_embed_parts.push(ref_feat.narrow(0, i, 1)?); // [1, 1, 2048]
    }

    // ref_audio_end position: text, no feat
    text_mask_parts.push(one.clone());
    feat_embed_parts.push(zero.clone());

    // Text token positions: text, no feat
    let n_text = text_ids.len();
    for _ in 0..n_text {
        text_mask_parts.push(one.clone());
        feat_embed_parts.push(zero.clone());
    }

    // Concatenate along dim 1 to get [1, total_len, 2048]
    let text_mask = Tensor::cat(&text_mask_parts, 1)?; // [1, total_len, 2048]
    let feat_embeds = Tensor::cat(&feat_embed_parts, 1)?; // [1, total_len, 2048]

    // ── 7. Build combined_embed ──
    // text_embed = embed_tokens(combined_ids) * scale_emb
    // Convert to F32 to match mask/feat_embeds dtype (text_mask is F32).
    // The AR function will convert to model dtype (BF16) before TSLM prefill.
    let combined_ids_t =
        Tensor::from_slice(&combined_ids, &[1, total_len], dev)?.to_dtype(DType::I64)?;
    let text_embed = tslm.embed_text(&combined_ids_t)?.to_dtype(DType::F32)?; // [1, total_len, 2048] F32

    // combined = text_mask * text_embed + (1 - text_mask) * feat_embeds
    let audio_mask = (text_mask.ones_like()? - &text_mask)?;
    let text_part = (text_mask * text_embed)?;
    let audio_part = (audio_mask * &feat_embeds)?;
    let combined_embeds = text_part.add(&audio_part)?;

    let elapsed = end_time.elapsed();
    eprintln!("  [clone] prefix ready: dims [1, {total_len}, 2048] in {elapsed:.2?}");

    Ok(RefPrefixResult {
        combined_ids: combined_ids_t,
        combined_embeds,
        feat_embeds,
        n_patches,
    })
}

fn build_voice_design_text(text: &str, voice_design: Option<&str>) -> String {
    match voice_design.map(str::trim).filter(|s| !s.is_empty()) {
        Some(control) => format!("({control}){text}"),
        None => text.to_string(),
    }
}

fn looks_like_traditional_chinese_hint(text: &str) -> bool {
    const TRADITIONAL_HINTS: &[char] = &[
        '這', '語', '聲', '請', '確', '認', '淨', '體', '測', '試', '雜', '會', '廣', '東', '國',
        '門', '開', '後', '應', '該', '聽', '說', '對', '齊', '產', '當', '無', '線', '電', '腦',
        '裡', '還', '點', '與', '為', '個', '們',
    ];
    text.chars().any(|ch| TRADITIONAL_HINTS.contains(&ch))
}

/// Quick helper: compute (mean, std, peak) of a tensor for debug tracing.
/// Save a tensor as raw f32 bytes for debug comparison.
#[cfg(feature = "debug-tensors")]
fn save_debug_tensor(t: &Tensor, name: &str) -> anyhow::Result<()> {
    let path = format!("output/{name}.f32");
    let flat = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&path, &bytes)?;
    Ok(())
}

fn tensor_stats(t: &Tensor) -> anyhow::Result<(f32, f32, f32)> {
    let flat = t.flatten_all()?.to_dtype(DType::F32)?;
    let n = flat.elem_count();
    let vals = flat.to_vec1::<f32>()?;
    let mean = vals.iter().sum::<f32>() / n as f32;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let peak = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    Ok((mean, var.sqrt(), peak))
}

/// Quick helper: compute peak of a tensor.
fn tensor_peak(t: &Tensor) -> anyhow::Result<f32> {
    Ok(t.flatten_all()?
        .to_dtype(DType::F32)?
        .abs()?
        .max_all()?
        .to_vec0::<f32>()?)
}

fn write_metrics_report(path: &Path, report: &SynthMetricsReport) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, format!("{json}\n"))?;
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn voice_design_text_matches_python_cli() {
        assert_eq!(build_voice_design_text("Hello", None), "Hello");
        assert_eq!(build_voice_design_text("Hello", Some("")), "Hello");
        assert_eq!(
            build_voice_design_text("Hello", Some(" warm female voice ")),
            "(warm female voice)Hello"
        );
    }

    #[test]
    fn traditional_chinese_hint_detects_mandarin_prompt_risk() {
        assert!(looks_like_traditional_chinese_hint("你好，這是語音測試。"));
        assert!(!looks_like_traditional_chinese_hint("你好，这是语音测试。"));
    }

    #[test]
    fn dry_run_writes_metrics_report() -> anyhow::Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let base = std::env::temp_dir().join(format!("voxcpm2-metrics-{unique}"));
        let wav_path = base.with_extension("wav");
        let metrics_path = base.with_extension("json");

        let mut pipe = VoxPipeline::new("cpu", None, true)?;
        let req = SynthRequest {
            text: "metrics smoke".into(),
            model_dir: None,
            output_path: wav_path.clone(),
            device: "cpu".into(),
            dry_run: true,
            metrics_output_path: Some(metrics_path.clone()),
            ..SynthRequest::default()
        };

        let result = pipe.synthesize(&req, None)?;
        assert!(wav_path.exists());
        assert!(metrics_path.exists());
        assert_eq!(result.output_path, wav_path);

        let metrics = std::fs::read_to_string(&metrics_path)?;
        let value: serde_json::Value = serde_json::from_str(&metrics)?;
        assert_eq!(value["dry_run"], true);
        assert_eq!(value["is_clone"], false);
        assert!(value["polish"].is_null());
        assert_eq!(value["samples"], result.samples);

        let _ = std::fs::remove_file(wav_path);
        let _ = std::fs::remove_file(metrics_path);
        Ok(())
    }

    /// Verify the full pipeline shapes with real model weights.
    /// Prerequisites:
    ///   1. model.safetensors in models/VoxCPM2/
    ///   2. audiovae.safetensors in models/VoxCPM2/
    ///   3. tokenizer files in models/VoxCPM2/
    #[test]
    #[ignore = "requires model.safetensors + audiovae.safetensors in models/VoxCPM2/"]
    fn test_full_pipeline_shapes() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let model_dir = if Path::new("models/VoxCPM2/model.safetensors").exists() {
            PathBuf::from("models/VoxCPM2")
        } else if Path::new("../../models/VoxCPM2/model.safetensors").exists() {
            PathBuf::from("../../models/VoxCPM2")
        } else {
            anyhow::bail!("model.safetensors not found");
        };

        let config = VoxConfig::load(&model_dir)?;
        let main_vb = weights::load_main_vb(&model_dir, &dev)?;
        let audiovae_tensors = weights::load_audiovae_decoder_tensors(&model_dir, &dev)?;
        let tokenizer = crate::tokenizer::VoxTokenizer::from_model_dir(&model_dir)?;

        // ── Tokenize ──
        let text = "Hello world.";
        let tokens = tokenizer.encode(text)?;
        assert!(!tokens.is_empty(), "tokenizer returned empty");
        let input_ids =
            Tensor::from_slice(&tokens, &[1, tokens.len()], &dev)?.to_dtype(DType::I64)?;
        let seq_len = tokens.len();
        println!("Tokens: {tokens:?}");

        // ── TSLM forward (28 layers) ──
        let mut tslm = TSLM::load(&main_vb.pp("base_lm"), &config.lm_config, &dev, false)?;
        let h_tslm = tslm.forward(&input_ids, 0)?;
        assert_eq!(h_tslm.dims(), &[1, seq_len, 2048], "TSLM output shape");
        println!("TSLM forward OK: {:?}", h_tslm.shape());

        // ── RALM forward (8 layers) ──
        let mut ralm = RALM::load(
            &main_vb.pp("residual_lm"),
            &config.lm_config,
            config.residual_lm_num_layers,
            &dev,
            false,
        )?;
        let h_ralm = ralm.forward(&h_tslm, 0)?;
        assert_eq!(h_ralm.dims(), &[1, seq_len, 2048], "RALM output shape");
        println!("RALM forward OK: {:?}", h_ralm.shape());

        // ── Text→DiT projection ──
        let (lm_to_dit, res_to_dit) = weights::load_text_to_dit_projections(&main_vb)?;
        let cond_text = (lm_to_dit.forward(&h_tslm)? + res_to_dit.forward(&h_ralm)?)?;
        assert_eq!(cond_text.dims(), &[1, seq_len, 1024], "cond_text shape");
        println!("Cond text OK: {:?}", cond_text.shape());
        // Debug cond_text stats
        let ct = cond_text.to_vec3::<f32>()?;
        let mut vals: Vec<f32> = ct
            .iter()
            .flat_map(|b| b.iter().flat_map(|t| t.iter()))
            .copied()
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        let ct_mean = vals.iter().sum::<f32>() / n as f32;
        let ct_std = (vals.iter().map(|v| (v - ct_mean).powi(2)).sum::<f32>() / n as f32).sqrt();
        println!(
            "  cond_text: mean={ct_mean:.6} std={ct_std:.6} p1={:.6} p99={:.6}",
            vals[n / 100],
            vals[99 * n / 100]
        );

        // ── LocDiT: diffusion generate ──
        // TODO(Z7): replace with autoregressive loop test
        let feat_dim = config.feat_dim;
        let latent = Tensor::zeros(&[1, feat_dim, seq_len], DType::F32, &dev)?;
        println!("  [test] WARNING: using zero latent (autoregressive loop not yet implemented)");

        // ── AudioVAE decode ──
        let vae = AudioVAE::load(&audiovae_tensors, &config.audio_vae_config)?;
        let waveform = vae.decode(&latent)?;
        let (n, c, samples) = waveform.shape().dims3()?;
        assert_eq!(n, 1, "AudioVAE batch dim");
        assert_eq!(c, 1, "AudioVAE mono");
        assert!(samples > 0, "AudioVAE produced empty output");
        println!("AudioVAE decode OK: [{n}, {c}, {samples}]");

        // Sanity check
        vae.check_audio(&waveform)?;
        println!("AudioVAE sanity check PASSED");

        // Check peak value
        let peak_v = waveform
            .to_dtype(candle_core::DType::F32)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        println!("AudioVAE peak: {peak_v:.6}");
        if peak_v < 1e-4 {
            println!("WARNING: very quiet output (peak={peak_v:.6}), may need gain");
        }

        println!("\n✓ Full pipeline shape verification PASSED");
        Ok(())
    }
}
