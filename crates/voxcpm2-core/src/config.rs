use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    pub model_dir: String,
    pub device: String,
    pub default_cfg: f32,
    pub default_steps: usize,
    pub output_dir: String,
    pub label_ai_generated: bool,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            model_dir: "models/VoxCPM2".to_string(),
            device: "auto".to_string(),
            default_cfg: 2.5,
            default_steps: 30,
            output_dir: "output".to_string(),
            label_ai_generated: true,
        }
    }
}

impl LocalConfig {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxConfig {
    pub architecture: String,
    pub lm_config: LmConfig,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub scalar_quantization_latent_dim: usize,
    pub scalar_quantization_scale: usize,
    pub residual_lm_num_layers: usize,
    pub residual_lm_no_rope: bool,
    pub encoder_config: EncoderConfig,
    pub dit_config: DitConfig,
    pub audio_vae_config: AudioVaeConfig,
    pub max_length: usize,
    pub device: Option<String>,
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmConfig {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub kv_channels: usize,
    pub vocab_size: usize,
    pub use_mup: bool,
    pub scale_emb: Option<f64>,
    pub dim_model_base: Option<f64>,
    pub scale_depth: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub scaling_type: String,
    pub short_factor: Vec<f64>,
    pub long_factor: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DitConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub kv_channels: usize,
    pub mean_mode: bool,
    pub cfm_config: CfmConfig,
    /// Optional latent normalization scale before AudioVAE decode.
    /// When set, the CFM latent is multiplied by this factor to better match
    /// the Python reference distribution (Rust std ~1.60 vs Python ~1.26).
    /// Recommended: 0.7875 (= 1.26 / 1.60) to match Python latent scale.
    #[serde(default)]
    pub latent_norm_scale: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfmConfig {
    pub sigma_min: f64,
    pub solver: String,
    pub t_scheduler: String,
    pub inference_cfg_rate: f64,
    /// Log-normal scheduler mean (default -1.0).
    #[serde(default = "default_lognorm_mean")]
    pub t_scheduler_mean: f64,
    /// Log-normal scheduler std (default 0.6).
    #[serde(default = "default_lognorm_std")]
    pub t_scheduler_std: f64,
}

fn default_lognorm_mean() -> f64 {
    -1.0
}
fn default_lognorm_std() -> f64 {
    0.6
}

impl Default for CfmConfig {
    fn default() -> Self {
        Self {
            sigma_min: 1e-6,
            solver: "euler".into(),
            t_scheduler: "uniform".into(),
            inference_cfg_rate: 2.5,
            t_scheduler_mean: -1.0,
            t_scheduler_std: 0.6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub sr_bin_boundaries: Vec<usize>,
    pub sample_rate: u32,
    pub out_sample_rate: u32,
}

impl VoxConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let text = fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_local_config_is_safe() {
        let cfg = LocalConfig::default();
        assert_eq!(cfg.device, "auto");
        assert!(cfg.label_ai_generated);
    }
}
