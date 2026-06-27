use crate::{
    audio,
    config::VoxConfig,
    device::{self, DevicePreference},
    models::*,
    weights,
};
use candle_core::{DType, Device, Module, Tensor};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
        } else if let Some(dir) = model_dir {
            Some(VoxConfig::load(dir)?)
        } else {
            None
        };
        Ok(Self { device, config })
    }

    pub fn synthesize(&mut self, req: &SynthRequest) -> anyhow::Result<SynthResult> {
        let sample_rate = self
            .config
            .as_ref()
            .map(|c| c.audio_vae_config.out_sample_rate)
            .unwrap_or(48_000);

        let samples = if req.dry_run {
            audio::smoke_tone(&req.text, sample_rate)
        } else {
            let model_dir = req
                .model_dir
                .as_deref()
                .unwrap_or_else(|| Path::new("models/VoxCPM2"));
            self.synthesize_real(req, model_dir, sample_rate)?
        };

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
    ) -> anyhow::Result<Vec<f32>> {
        let dev = &self.device;
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("VoxConfig required for real synthesis"))?;

        // ── Load all weights ──
        let main_vb = weights::load_main_vb(model_dir, dev)?;
        let audiovae_tensors = weights::load_audiovae_decoder_tensors(model_dir, dev)?;

        // ── Tokenizer ──
        let tokenizer = crate::tokenizer::VoxTokenizer::from_model_dir(model_dir)?;
        let tokens = tokenizer.encode(&req.text)?;
        if tokens.is_empty() {
            anyhow::bail!("tokenizer returned empty tokens");
        }
        let input_ids =
            Tensor::from_slice(&tokens, &[1, tokens.len()], dev)?.to_dtype(DType::I64)?;
        let _seq_len = tokens.len();

        // ── TSLM forward ──
        let mut tslm = TSLM::load(&main_vb.pp("base_lm"), &config.lm_config, dev, false)?;
        let h_tslm = tslm.forward(&input_ids, 0)?; // [1, seq_len, 2048]

        // ── RALM forward ──
        let ralm_num_layers = config.residual_lm_num_layers;
        let mut ralm = RALM::load(
            &main_vb.pp("residual_lm"), &config.lm_config,
            ralm_num_layers, dev, false,
        )?;
        let h_ralm = ralm.forward(&h_tslm, 0)?; // [1, seq_len, 2048]

        // ── Text → DiT condition (1024-dim) ──
        let (lm_to_dit, res_to_dit) = weights::load_text_to_dit_projections(&main_vb)?;
        let cond_text = (lm_to_dit.forward(&h_tslm)? + res_to_dit.forward(&h_ralm)?)?; // [1, seq_len, 1024]

        // ── LocDiT: diffusion generate ──
        let feat_dim = config.feat_dim; // 64
        let sched = FlowMatchingScheduler::from_config(&config.dit_config.cfm_config);
        let mut dit = LocDiT::load(
            &main_vb.pp("feat_decoder"),
            &config.dit_config,
            feat_dim,
            sched,
        )?;

        // Mean-pool 1024→64 for DiT cond (placeholder until proper fix)
        let (b, t, _) = cond_text.shape().dims3()?;
        let group_size = 1024 / 64;
        let cond_dit = cond_text
            .reshape((b, t, 64, group_size))?
            .mean(3)?;

        let latent = dit.generate(&cond_dit, req.inference_timesteps, Some(42))?;
        // latent shape: [1, feat_dim, seq_len]
        let latent_peak = latent.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
        if latent_peak < 0.001 {
            tracing::warn!("DiT latent very quiet (peak={:.6})", latent_peak);
        }

        // ── AudioVAE decode ──
        let vae = crate::models::AudioVAE::load(&audiovae_tensors, &config.audio_vae_config)?;
        let waveform = vae.decode(&latent)?;
        // waveform shape: [1, 1, samples]

        // Flatten to Vec<f32>
        let vals = waveform.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vals)
    }
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
        let feat_dim = config.feat_dim;
        let sched = FlowMatchingScheduler::from_config(&config.dit_config.cfm_config);
        let mut dit = LocDiT::load(
            &main_vb.pp("feat_decoder"),
            &config.dit_config,
            feat_dim,
            sched,
        )?;
        // Pass 1024-dim cond directly (DiT skips cond_proj when dim == hidden_dim)
        let cond_dit = cond_text
            .reshape((1, seq_len, 64, 16))?
            .mean(3)?;
        assert_eq!(cond_dit.dims(), &[1, seq_len, 64],
            "cond_dit shape before LocDiT (mean-pooled to 64)");
        // Debug cond_dit stats
        let cd = cond_dit.to_vec3::<f32>()?;
        let mut cd_vals: Vec<f32> = cd.iter().flat_map(|b| b.iter().flat_map(|t| t.iter())).copied().collect();
        cd_vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let cn = cd_vals.len();
        let cd_mean = cd_vals.iter().sum::<f32>() / cn as f32;
        let cd_std = (cd_vals.iter().map(|v| (v - cd_mean).powi(2)).sum::<f32>() / cn as f32).sqrt();
        println!("  cond_dit (mean-pooled 1024→64): mean={cd_mean:.6} std={cd_std:.6} p1={:.6} p99={:.6}",
            cd_vals[cn/100], cd_vals[99*cn/100]);
        let latent = dit.generate(&cond_dit, 2, Some(42))?;
        assert_eq!(latent.dims(), &[1, feat_dim, seq_len],
            "LocDiT latent shape");
        println!("LocDiT generate OK: {:?}", latent.shape());
        // Debug latent stats
        let lt = latent.to_vec3::<f32>()?;
        let mut lt_vals: Vec<f32> = lt.iter().flat_map(|b| b.iter().flat_map(|t| t.iter())).copied().collect();
        lt_vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let ln = lt_vals.len();
        let lt_mean = lt_vals.iter().sum::<f32>() / ln as f32;
        let lt_std = (lt_vals.iter().map(|v| (v - lt_mean).powi(2)).sum::<f32>() / ln as f32).sqrt();
        println!("  latent: mean={lt_mean:.6} std={lt_std:.6} p1={:.6} p99={:.6} peak={:.6}",
            lt_vals[ln/100], lt_vals[99*ln/100], lt_vals[ln-1]);

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


