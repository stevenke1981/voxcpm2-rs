use crate::{
    audio,
    config::VoxConfig,
    device::{self, DevicePreference},
};
use candle_core::Device;
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
            // 實作順序：tokenizer → TSLM → RALM → LocDiT → AudioVAE。
            // 這裡故意 fail fast，避免假裝已完成真實 VoxCPM2 推理。
            anyhow::bail!("real VoxCPM2 inference is not implemented yet; run with --dry-run or complete docs/todos.md milestones C-G")
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
}
