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
        let mut ralm = RALM::load(&main_vb.pp("residual_lm"), &config.lm_config, dev, false)?;
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

        // The DiT cond_proj expects [batch, time, feat_dim] (64-dim input).
        // The 1024-dim text cond needs further processing to 64-dim.
        // For now, use a mean-pooled + projected approximation.
        // TODO: Replace with proper feat_encoder or learned projection once
        // the feature encoder architecture is validated.
        let cond_dit = project_cond_to_feat_dim(&cond_text, feat_dim)?;

        let latent = dit.generate(&cond_dit, req.inference_timesteps, Some(42))?;
        // latent shape: [1, feat_dim, seq_len]

        // ── AudioVAE decode ──
        let vae = crate::models::AudioVAE::load(&audiovae_tensors, &config.audio_vae_config)?;
        let waveform = vae.decode(&latent)?;
        // waveform shape: [1, 1, samples]

        // Flatten to Vec<f32>
        let vals = waveform.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vals)
    }
}

/// Temporary projection from 1024-dim text cond to 64-dim feat_dim cond.
///
/// The real VoxCPM2 pipeline likely uses feat_encoder or a learned projection
/// to reduce dimensionality. For now we apply mean pooling across heads.
fn project_cond_to_feat_dim(cond: &Tensor, feat_dim: usize) -> anyhow::Result<Tensor> {
    let (b, t, _hidden) = cond.shape().dims3()?;
    // Simple learned projection via matmul (we don't have a dedicated weight,
    // so create a random fixed projection as placeholder)
    // Averages over hidden/feat_dim groups
    let group_size = _hidden / feat_dim;
    if _hidden % feat_dim == 0 {
        // Mean pool: reshape [b, t, hidden] → [b, t, feat_dim, group] → mean along last dim
        let reshaped = cond.reshape((b, t, feat_dim, group_size))?;
        let pooled = reshaped.mean(3)?;
        Ok(pooled)
    } else {
        // Fallback: slice to feat_dim
        Ok(cond.narrow(2, 0, feat_dim)?)
    }
}
