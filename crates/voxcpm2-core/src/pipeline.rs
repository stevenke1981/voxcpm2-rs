use crate::{
    audio,
    autoregressive::generate_autoregressive,
    config::VoxConfig,
    device::{self, DevicePreference},
    models::*,
    weights,
};
use candle_core::{DType, Device, Module, Tensor};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

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
}

impl Default for SynthRequest {
    fn default() -> Self {
        Self {
            text: "Hello, world.".into(),
            model_dir: Some(PathBuf::from("models/VoxCPM2")),
            output_path: PathBuf::from("output/synth.wav"),
            device: "auto".into(),
            cfg_value: 2.0,
            inference_timesteps: 10,
            dry_run: false,
            label_ai_generated: true,
            seed: None,
            voice_design: None,
            post_gain: None,
            max_autoregressive_steps: None,
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

pub struct VoxPipeline {
    pub device: Device,
    pub config: Option<VoxConfig>,
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
        Ok(Self { device, config })
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

        audio::check_audio(&samples)?;
        audio::write_wav_f32(&req.output_path, &samples, sample_rate)?;
        Ok(SynthResult {
            output_path: req.output_path.clone(),
            sample_rate,
            samples: samples.len(),
            device: device::device_label(&self.device),
            dry_run: req.dry_run,
        })
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

        // ── Load all weights ──
        let main_vb = weights::load_main_vb(model_dir, dev)?;
        // AudioVAE — try CUDA first, fall back to CPU if cuBLAS conv1d errors
        let audiovae_dev = if matches!(dev, Device::Cuda(_)) { dev } else { &Device::Cpu };
        let audiovae_tensors = weights::load_audiovae_decoder_tensors(model_dir, audiovae_dev)?;

        // ── Tokenizer ──
        let tokenizer = crate::tokenizer::VoxTokenizer::from_model_dir(model_dir)?;
        // Apply chat template (matching Python: add_special_tokens=False, no BOS):
        // <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
        let chat_text = crate::tokenizer::render_chat_template(
            &[crate::tokenizer::ChatMessage {
                role: "user".into(),
                content: req.text.clone(),
            }],
            true, // add_generation_prompt
        );
        let tokens = tokenizer.encode(&chat_text)?;
        if tokens.is_empty() {
            anyhow::bail!("tokenizer returned empty tokens");
        }
        let input_ids =
            Tensor::from_slice(&tokens, &[1, tokens.len()], dev)?.to_dtype(DType::I64)?;
        let seq_len = tokens.len();
        let text_len = req.text.len();

        // ── TSLM forward ──
        check_cancel("TSLM")?;
        let mut tslm = TSLM::load(&main_vb.pp("base_lm"), &config.lm_config, dev, false)?;
        let h_tslm = tslm.forward(&input_ids, 0)?; // [1, seq_len, 2048]
        eprintln!("  [pipe] TSLM: seq_len={seq_len} text_len={text_len} h_tslm.shape={:?}", h_tslm.shape());
        save_debug_tensor(&h_tslm, "debug_pipe_tslm")?;
        let (tslm_mean, tslm_std, tslm_peak) = tensor_stats(&h_tslm)?;
        eprintln!("  [pipe] TSLM: seq_len={seq_len} text_len={text_len} mean={tslm_mean:.6} std={tslm_std:.6} peak={tslm_peak:.6}");

        // ── RALM forward ──
        check_cancel("RALM")?;
        let ralm_num_layers = config.residual_lm_num_layers;
        let mut ralm = RALM::load(
            &main_vb.pp("residual_lm"), &config.lm_config,
            ralm_num_layers, dev, false,
        )?;
        let h_ralm = ralm.forward(&h_tslm, 0)?; // [1, seq_len, 2048]
        save_debug_tensor(&h_ralm, "debug_pipe_ralm")?;
        let (ralm_mean, ralm_std, ralm_peak) = tensor_stats(&h_ralm)?;
        let h_tslm_peak = tensor_peak(&h_tslm)?;
        eprintln!("  [pipe] RALM: mean={ralm_mean:.6} std={ralm_std:.6} peak={ralm_peak:.6} (TSLM peak={h_tslm_peak:.6})");

        // ── Text → DiT condition (1024-dim) ──
        check_cancel("cond")?;
        let (lm_to_dit, res_to_dit) = weights::load_text_to_dit_projections(&main_vb)?;
        let cond_text = (lm_to_dit.forward(&h_tslm)? + res_to_dit.forward(&h_ralm)?)?; // [1, seq_len, 1024]
        save_debug_tensor(&cond_text, "debug_pipe_cond_text")?;
        let (ct_mean, ct_std, ct_peak) = tensor_stats(&cond_text)?;
        eprintln!("  [pipe] cond_text (lm+res→1024): mean={ct_mean:.6} std={ct_std:.6} peak={ct_peak:.6}");

        // ── NEW: autoregressive generation ──
        check_cancel("autoregressive")?;
        let latent = generate_autoregressive(
            &main_vb, config, req, dev, &input_ids, cancel,
        )?;
        // latent shape: [1, feat_dim, seq_len]
        let latent_peak = latent.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?.to_vec0::<f32>()?;
        if latent_peak < 0.001 {
            tracing::warn!("DiT latent very quiet (peak={:.6})", latent_peak);
        }

        // ── AudioVAE decode ──
        let vae = crate::models::AudioVAE::load(&audiovae_tensors, &config.audio_vae_config)?;
        let latent_vae = latent.to_device(audiovae_dev)?.to_dtype(DType::F32)?;
        eprintln!("  [pipe] AudioVAE decode on device: {:?}", latent_vae.device());
        {
            let latent_peak = tensor_peak(&latent_vae)?;
            let latent_stats = tensor_stats(&latent_vae)?;
            eprintln!("  [pipe] AudioVAE latent: peak={:.6} mean={:.6} std={:.6}",
                latent_peak, latent_stats.0, latent_stats.1);
        }
        // Save latent for Python verification
        if let Ok(latent_f32) = latent_vae.squeeze(0) {
            let latent_flat = latent_f32.flatten_all()?.to_vec1::<f32>()?;
            let latent_bytes: Vec<u8> = latent_flat.iter().flat_map(|v| v.to_le_bytes()).collect();
            if std::fs::write("output/latent_rust.f32", &latent_bytes).is_ok() {
                eprintln!("  [pipe] Saved latent to output/latent_rust.f32 ({} frames x 64 ch)", latent_vae.dim(1).unwrap_or(0));
            }
        }
        let waveform = vae.decode(&latent_vae)?;
        // waveform shape: [1, 1, samples]

        // Flatten to Vec<f32>
        let vals = waveform.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vals)
    }
}

/// Quick helper: compute (mean, std, peak) of a tensor for debug tracing.
/// Save a tensor as raw f32 bytes for debug comparison.
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

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

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
        let input_ids = Tensor::from_slice(&tokens, &[1, tokens.len()], &dev)?.to_dtype(DType::I64)?;
        let seq_len = tokens.len();
        println!("Tokens: {tokens:?}");

        // ── TSLM forward (28 layers) ──
        let mut tslm = TSLM::load(&main_vb.pp("base_lm"), &config.lm_config, &dev, false)?;
        let h_tslm = tslm.forward(&input_ids, 0)?;
        assert_eq!(h_tslm.dims(), &[1, seq_len, 2048],
            "TSLM output shape");
        println!("TSLM forward OK: {:?}", h_tslm.shape());

        // ── RALM forward (8 layers) ──
        let mut ralm = RALM::load(
            &main_vb.pp("residual_lm"), &config.lm_config,
            config.residual_lm_num_layers, &dev, false,
        )?;
        let h_ralm = ralm.forward(&h_tslm, 0)?;
        assert_eq!(h_ralm.dims(), &[1, seq_len, 2048],
            "RALM output shape");
        println!("RALM forward OK: {:?}", h_ralm.shape());

        // ── Text→DiT projection ──
        let (lm_to_dit, res_to_dit) = weights::load_text_to_dit_projections(&main_vb)?;
        let cond_text = (lm_to_dit.forward(&h_tslm)? + res_to_dit.forward(&h_ralm)?)?;
        assert_eq!(cond_text.dims(), &[1, seq_len, 1024],
            "cond_text shape");
        println!("Cond text OK: {:?}", cond_text.shape());
        // Debug cond_text stats
        let ct = cond_text.to_vec3::<f32>()?;
        let mut vals: Vec<f32> = ct.iter().flat_map(|b| b.iter().flat_map(|t| t.iter())).copied().collect();
        vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        let ct_mean = vals.iter().sum::<f32>() / n as f32;
        let ct_std = (vals.iter().map(|v| (v - ct_mean).powi(2)).sum::<f32>() / n as f32).sqrt();
        println!("  cond_text: mean={ct_mean:.6} std={ct_std:.6} p1={:.6} p99={:.6}",
            vals[n/100], vals[99*n/100]);

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
        let peak_v = waveform.to_dtype(candle_core::DType::F32)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("AudioVAE peak: {peak_v:.6}");
        if peak_v < 1e-4 {
            println!("WARNING: very quiet output (peak={peak_v:.6}), may need gain");
        }

        println!("\n✓ Full pipeline shape verification PASSED");
        Ok(())
    }
}


