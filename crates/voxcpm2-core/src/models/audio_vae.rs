//! AudioVAE V2 Decoder — faithful reimplementation matching real model structure.
//!
//! Decoder tensor structure (from audiovae.safetensors, weight_norm fused):
//!
//! model.0: depthwise Conv1d(64→64, k=7, groups=64) + bias   [in=1→out=64 per group]
//! model.1: pointwise Conv1d(64→2048, k=1) + bias
//! model.2–7: UpsamplingResidualBlock each:
//!   block.0: alpha [1, C_in, 1] (learned scale)
//!   block.1: ConvTranspose1d(C_in→C_out, kernel=K, stride=S) — upsampling
//!   block.2–4: ResidualSubBlock(C_out) each:
//!     block.0: alpha [1, C, 1]
//!     block.1: depthwise Conv1d(C→C, k=7, groups=C) + bias
//!     block.2: alpha [1, C, 1]
//!     block.3: pointwise Conv1d(C→C, k=1) + bias
//! model.8: alpha [1, 32, 1]
//! model.9: Conv1d(32→1, k=7) + bias   — final output
//!
//! Upsample rates: [8, 6, 5, 2, 2, 2] → total 960×
//! Kernel matches: model.2(16→8) model.3(12→6) model.4(10→5) model.5-7(4→2)

use crate::config::AudioVaeConfig;
use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply SiLU activation.
fn silu(x: &Tensor) -> Result<Tensor> {
    Activation::Silu.forward(x)
}

/// Load a 1D tensor by name from the raw map and squeeze to 1D if needed.
fn load_1d(map: &HashMap<String, Tensor>, name: &str, len: usize) -> Result<Tensor> {
    let t = map
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    if t.shape().dims() == [1, len, 1] {
        // alpha tensor: [1, C, 1] → squeeze to [C]
        t.squeeze(0)?.squeeze(1)
    } else {
        Ok(t.clone())
    }
}

/// Load a 3D conv weight (fused weight_norm) by name.
fn load_conv_weight(
    map: &HashMap<String, Tensor>,
    name: &str,
    shape: &[usize; 3],
) -> Result<Tensor> {
    let t = map
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    let actual: Vec<usize> = t.shape().dims().to_vec();
    assert_eq!(
        actual, shape,
        "shape mismatch for {name}: expected {shape:?}, got {actual:?}"
    );
    Ok(t.clone())
}

// ---------------------------------------------------------------------------
// Alpha — learned per-channel scale  [1, C, 1]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Alpha {
    scale: Tensor, // [C] stored as 1D, broadcast as [1, C, 1]
}

impl Alpha {
    fn load(map: &HashMap<String, Tensor>, name: &str, channels: usize) -> Result<Self> {
        let scale = load_1d(map, name, channels)?;
        Ok(Self { scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [N, C, L]; scale: [C] → broadcast to [1, C, 1]
        let s = self.scale.reshape((1, self.scale.dim(0)?, 1))?;
        x.broadcast_mul(&s)
    }
}

// ---------------------------------------------------------------------------
// ResidualSubBlock — alpha → depthwise Conv1d → alpha → pointwise Conv1d + skip
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ResidualSubBlock {
    alpha0: Alpha,
    depthwise: Conv1d, // groups=C
    alpha1: Alpha,
    pointwise: Conv1d, // k=1
}

impl ResidualSubBlock {
    fn load(map: &HashMap<String, Tensor>, prefix: &str, channels: usize) -> Result<Self> {
        let alpha0 = Alpha::load(map, &format!("{prefix}.block.0.alpha"), channels)?;

        let dw_weight =
            load_conv_weight(map, &format!("{prefix}.block.1.weight"), &[channels, 1, 7])?;
        let dw_bias = load_1d(map, &format!("{prefix}.block.1.bias"), channels)?;
        let depthwise = Conv1d::new(
            dw_weight,
            Some(dw_bias),
            Conv1dConfig {
                padding: 3,       // (7-1)/2 = same
                groups: channels, // depthwise
                ..Default::default()
            },
        );

        let alpha1 = Alpha::load(map, &format!("{prefix}.block.2.alpha"), channels)?;

        let pw_weight = load_conv_weight(
            map,
            &format!("{prefix}.block.3.weight"),
            &[channels, channels, 1],
        )?;
        let pw_bias = load_1d(map, &format!("{prefix}.block.3.bias"), channels)?;
        let pointwise = Conv1d::new(
            pw_weight,
            Some(pw_bias),
            Conv1dConfig {
                padding: 0,
                ..Default::default()
            },
        );

        Ok(Self {
            alpha0,
            depthwise,
            alpha1,
            pointwise,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.alpha0.forward(x)?;
        let h = silu(&self.depthwise.forward(&h)?)?;
        let h = self.alpha1.forward(&h)?;
        let h = silu(&self.pointwise.forward(&h)?)?;
        h + residual
    }
}

// ---------------------------------------------------------------------------
// UpsamplingBlock — alpha → ConvTranspose1d → 3× ResidualSubBlock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct UpsamplingBlock {
    alpha: Alpha,
    up_conv: ConvTranspose1d,
    sub_blocks: Vec<ResidualSubBlock>,
}

impl UpsamplingBlock {
    fn load(
        map: &HashMap<String, Tensor>,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let alpha = Alpha::load(map, &format!("{prefix}.block.0.alpha"), c_in)?;

        // ConvTranspose1d weight loaded from fused weight_norm:
        // weight shape: [in_channels, out_channels, kernel_size] = [c_in, c_out, kernel]
        let up_weight = load_conv_weight(
            map,
            &format!("{prefix}.block.1.weight"),
            &[c_in, c_out, kernel],
        )?;
        // bias shape: [out_channels] = [c_out]
        let up_bias = load_1d(map, &format!("{prefix}.block.1.bias"), c_out)?;

        // ConvTranspose1d padding for exact output L_out = stride * L_in:
        //   L_out = (L_in-1)*stride + kernel - 2*padding + output_padding
        //   => padding = ceil((kernel - stride) / 2)
        //   => output_padding = stride - kernel + 2*padding  (always < stride, >= 0)
        let padding = if kernel > stride {
            (kernel - stride).div_ceil(2)
        } else {
            0
        };
        let op = stride as isize - kernel as isize + 2 * padding as isize;
        let output_padding = if op >= 0 { op as usize } else { 0 };
        debug_assert!(
            output_padding < stride,
            "output_padding {output_padding} >= stride {stride}"
        );

        let up_conv = ConvTranspose1d::new(
            up_weight,
            Some(up_bias),
            ConvTranspose1dConfig {
                padding,
                output_padding,
                stride,
                ..Default::default()
            },
        );

        let mut sub_blocks = Vec::new();
        for i in 2..=4 {
            let sub = ResidualSubBlock::load(map, &format!("{prefix}.block.{i}"), c_out)?;
            sub_blocks.push(sub);
        }

        Ok(Self {
            alpha,
            up_conv,
            sub_blocks,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.alpha.forward(x)?;
        let h = silu(&self.up_conv.forward(&h)?)?;
        let mut h = h;
        for sb in &self.sub_blocks {
            h = sb.forward(&h)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// AudioVAE Decoder
// ---------------------------------------------------------------------------

pub struct AudioVAE {
    model_0: Conv1d,                 // depthwise 64→64, k=7
    model_1: Conv1d,                 // pointwise 64→2048, k=1
    up_blocks: Vec<UpsamplingBlock>, // model 2–7
    model_8: Alpha,                  // alpha [1, 32, 1]
    model_9: Conv1d,                 // 32→1, k=7
    pub sample_rate: u32,
    pub out_sample_rate: u32,
}

impl AudioVAE {
    /// Load decoder from the raw fused tensor map.
    ///
    /// The map must contain `decoder.model.N.*` tensors with weight_norm already fused.
    /// Use `weights::load_audiovae_decoder_tensors()` to obtain it.
    pub fn load(tensors: &HashMap<String, Tensor>, cfg: &AudioVaeConfig) -> Result<Self> {
        // Strip the "decoder." prefix — the map has keys like "decoder.model.0.weight"
        // We'll use the full keys directly.

        // model.0: depthwise Conv1d(64, 64, k=7, groups=64)
        let w0 = load_conv_weight(tensors, "decoder.model.0.weight", &[64, 1, 7])?;
        let b0 = load_1d(tensors, "decoder.model.0.bias", 64)?;
        let model_0 = Conv1d::new(
            w0,
            Some(b0),
            Conv1dConfig {
                padding: 3,
                groups: 64,
                ..Default::default()
            },
        );

        // model.1: pointwise Conv1d(64, 2048, k=1)
        let w1 = load_conv_weight(tensors, "decoder.model.1.weight", &[2048, 64, 1])?;
        let b1 = load_1d(tensors, "decoder.model.1.bias", 2048)?;
        let model_1 = Conv1d::new(
            w1,
            Some(b1),
            Conv1dConfig {
                padding: 0,
                ..Default::default()
            },
        );

        // model.2–7: Upsampling blocks
        let up_specs: [(usize, usize, usize, usize); 6] = [
            (2, 2048, 1024, 16), // rate=8
            (3, 1024, 512, 12),  // rate=6
            (4, 512, 256, 10),   // rate=5
            (5, 256, 128, 4),    // rate=2
            (6, 128, 64, 4),     // rate=2
            (7, 64, 32, 4),      // rate=2
        ];
        let rates = &cfg.decoder_rates; // [8, 6, 5, 2, 2, 2]
        let mut up_blocks = Vec::new();
        for (idx, &(n, c_in, c_out, kernel)) in up_specs.iter().enumerate() {
            let stride = rates[idx];
            let block = UpsamplingBlock::load(
                tensors,
                &format!("decoder.model.{n}"),
                c_in,
                c_out,
                kernel,
                stride,
            )?;
            up_blocks.push(block);
        }

        // model.8: alpha [1, 32, 1]
        let model_8 = Alpha::load(tensors, "decoder.model.8.alpha", 32)?;

        // model.9: Conv1d(32, 1, k=7)
        let w9 = load_conv_weight(tensors, "decoder.model.9.weight", &[1, 32, 7])?;
        let b9 = load_1d(tensors, "decoder.model.9.bias", 1)?;
        let model_9 = Conv1d::new(
            w9,
            Some(b9),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );

        Ok(Self {
            model_0,
            model_1,
            up_blocks,
            model_8,
            model_9,
            sample_rate: cfg.sample_rate,
            out_sample_rate: cfg.out_sample_rate,
        })
    }

    /// Decode latent to waveform.
    ///
    /// # Args
    /// * `latent` — shape `[batch, 64, time]` — the 64-dim latent from DiT
    ///
    /// # Returns
    /// * shape `[batch, 1, samples]` — mono audio at `out_sample_rate`
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let mut h = latent.clone(); // [N, 64, T]

        // model.0: depthwise conv (no activation before first conv)
        h = silu(&self.model_0.forward(&h)?)?;

        // model.1: pointwise expansion
        h = silu(&self.model_1.forward(&h)?)?;

        // model.2–7: upsampling blocks
        for block in &self.up_blocks {
            h = block.forward(&h)?;
        }

        // model.8: alpha scale
        h = self.model_8.forward(&h)?;

        // model.9: final output conv
        h = self.model_9.forward(&h)?;
        // No activation on final output (raw waveform)
        Ok(h)
    }

    /// Check waveform sanity (peak, NaN, duration).
    pub fn check_audio(&self, waveform: &Tensor) -> Result<()> {
        let vals = waveform.flatten_all()?.to_vec1::<f32>()?;
        super::super::audio::check_audio(&vals)
            .map_err(|e| candle_core::Error::Msg(format!("AudioVAE check: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights;
    use candle_core::DType;

    #[test]
    fn audiovae_smoke() -> Result<()> {
        // Verify config construct
        let _cfg = AudioVaeConfig {
            encoder_dim: 128,
            encoder_rates: vec![2, 5, 8, 8],
            latent_dim: 64,
            decoder_dim: 2048,
            decoder_rates: vec![8, 6, 5, 2, 2, 2],
            sr_bin_boundaries: vec![20000, 30000, 40000],
            sample_rate: 16000,
            out_sample_rate: 48000,
        };
        Ok(())
    }

    #[test]
    #[ignore = "requires audiovae.safetensors in models/VoxCPM2/"]
    fn test_audiovae_decode_shape() -> Result<()> {
        let dev = candle_core::Device::Cpu;
        let model_dir = if std::path::Path::new("models/VoxCPM2").exists() {
            std::path::Path::new("models/VoxCPM2").to_path_buf()
        } else {
            std::path::Path::new("../../models/VoxCPM2").to_path_buf()
        };
        // Load only the audio_vae config from JSON (not the full VoxConfig to avoid anyhow)
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::Msg(format!("read config: {e}")))?;
        let full_cfg: crate::config::VoxConfig = serde_json::from_str(&config_text)
            .map_err(|e| candle_core::Error::Msg(format!("parse config: {e}")))?;
        let tensors = weights::load_audiovae_decoder_tensors(&model_dir, &dev)?;
        let vae = AudioVAE::load(&tensors, &full_cfg.audio_vae_config)?;

        // Create dummy latent: [1, 64, 16]
        let latent = Tensor::zeros(&[1, 64, 16], DType::F32, &dev)?;
        let waveform = vae.decode(&latent)?;

        // Should produce output: [1, 1, ~16*960 = 15360 samples]
        let (n, c, samples) = waveform.shape().dims3()?;
        assert_eq!(n, 1, "batch dim");
        assert_eq!(c, 1, "mono audio");
        assert!(samples > 15000, "expected ~15360 samples, got {samples}");
        println!("AudioVAE decode shape OK: [{n}, {c}, {samples}]");
        Ok(())
    }
}
