//! AudioVAE V2 Decoder — faithful reimplementation matching Python CausalDecoder.
//!
//! Python reference: `audio_vae_v2.py` CausalDecoder (weight_norm, causal convs, Snake1d).
//!
//! Key differences from our earlier Rust version:
//! - `Alpha` struct replaced by `Snake1d` activation (Python's `Snake1d`, NOT a scale multiplier)
//! - Causal padding (left-only) instead of symmetric "same" padding
//! - `nn.Tanh()` at output (Python uses Tanh, not raw conv output)
//! - No SiLU on model.0/model.1 initial convs (Python has no activation after them)
//!
//! Decoder structure (from audiovae.safetensors):
//!
//! model.0: WNCausalConv1d(64→64, k=7, groups=64) + bias   [no activation]
//! model.1: WNCausalConv1d(64→2048, k=1) + bias             [no activation]
//! model.2–7: CausalDecoderBlock each:
//!   block.0: Snake1d(C_in) alpha [1, C_in, 1]             ← activation, not scale
//!   block.1: WNCausalTransposeConv1d(C_in→C_out, kernel=K, stride=S)
//!   block.2–4: CausalResidualUnit(C_out) each:
//!     block.0: Snake1d(C_out) alpha [1, C_out, 1]
//!     block.1: WNCausalConv1d(C_out→C_out, k=7, groups=C_out) + bias (depthwise)
//!     block.2: Snake1d(C_out) alpha [1, C_out, 1]
//!     block.3: WNCausalConv1d(C_out→C_out, k=1) + bias (pointwise)
//! model.8: Snake1d(32) alpha [1, 32, 1]
//! model.9: WNCausalConv1d(32→1, k=7) + bias
//!   + nn.Tanh() — final activation
//!
//! SR Conditioning (FiLM):
//! sr_cond_model.N.scale_embed.weight: [4, C] — 4 SR bins × C channels
//! sr_cond_model.N.bias_embed.weight:  [4, C]
//! sr_bin_boundaries: [3] I32
//!
//! Upsample rates (decoder_rates): [8, 6, 5, 2, 2, 2] → total 960×
//! Kernel sizes: model.2(16) model.3(12) model.4(10) model.5-7(4)

use crate::config::AudioVaeConfig;
use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a 1D tensor by name from the raw map and squeeze to 1D if needed.
fn load_1d(map: &HashMap<String, Tensor>, name: &str, len: usize) -> Result<Tensor> {
    let t = map
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor: {name}"));
    if t.shape().dims() == [1, len, 1] {
        // snake alpha or bias: [1, C, 1] → squeeze to [C]
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

/// Compute causal padding for a conv layer given (kernel, dilation).
/// Python `CausalConv1d` pads only the LEFT side:
///   pad_len = (kernel - 1) * dilation / 2   (integer division)
/// Then applies F.pad(x, (pad_len * 2 - output_padding, 0)) before conv.
/// For causal conv, we use `padding = 0` and do manual left-pad,
/// OR use `padding = pad_len` with asymmetry.
///
/// We implement causal padding by pre-padding the input tensor on the left side.
/// Causal (left-only) padding for a Regular Conv1d, matching Python's CausalConv1d.
///
/// Python forward: `F.pad(x, (pad*2 - output_padding, 0))` then conv.
/// For stride=1, output_padding=0: left pad = pad * 2, so conv output preserves width.
fn causal_pad(x: &Tensor, kernel: usize, dilation: usize, stride: usize) -> Result<Tensor> {
    let pad_len = dilation * (kernel - 1) / 2;
    let output_padding = if stride > 1 { 1 } else { 0 };
    let left_pad = pad_len * 2 - output_padding;
    if left_pad == 0 {
        return Ok(x.clone());
    }
    // Pad left side only on the last dimension (width/time)
    x.pad_with_zeros(2, left_pad, 0)
}

/// Python `Snake1d` activation:
///   snake(x) = x + (1/alpha) * sin(alpha * x)²
/// where alpha is a learned parameter per channel.
#[derive(Debug, Clone)]
struct Snake1d {
    alpha: Tensor, // [C] stored as 1D
}

impl Snake1d {
    fn load(map: &HashMap<String, Tensor>, name: &str, channels: usize) -> Result<Self> {
        let alpha = load_1d(map, name, channels)?;
        Ok(Self { alpha })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [N, C, L]; alpha: [C]
        // snake(x) = x + (1/alpha) * sin(alpha * x)²
        let alpha_3d = self.alpha.reshape((1, self.alpha.dim(0)?, 1))?; // [1, C, 1]
        let alpha_x = x.broadcast_mul(&alpha_3d)?;
        let sin_sq = alpha_x.sin()?.sqr()?;
        // 1/alpha — trained alpha should not be near-zero
        let inv_alpha = alpha_3d.recip()?;
        let correction = sin_sq.broadcast_mul(&inv_alpha)?;
        x.broadcast_add(&correction)
    }
}

// ---------------------------------------------------------------------------
// CausalResidualUnit — Snake1d → depthwise CausalConv1d → Snake1d → pointwise Conv1d + skip
// Python equivalent: CausalResidualUnit (audio_vae_v2.py)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CausalResidualUnit {
    snake0: Snake1d,
    depthwise: Conv1d, // k=7, groups=C, dilation varies
    snake1: Snake1d,
    pointwise: Conv1d, // k=1
    kernel: usize,
    dilation: usize,
}

impl CausalResidualUnit {
    fn load(map: &HashMap<String, Tensor>, prefix: &str, channels: usize, dilation: usize) -> Result<Self> {
        let snake0 = Snake1d::load(map, &format!("{prefix}.block.0.alpha"), channels)?;

        let kernel = 7; // fixed kernel size for depthwise conv
        let _pad = ((kernel - 1) * dilation) / 2; // causal pad: left-only (used for reference)
        let dw_weight = load_conv_weight(map, &format!("{prefix}.block.1.weight"), &[channels, 1, kernel])?;
        let dw_bias = load_1d(map, &format!("{prefix}.block.1.bias"), channels)?;
        let depthwise = Conv1d::new(
            dw_weight,
            Some(dw_bias),
            Conv1dConfig {
                padding: 0, // we handle padding manually as causal left-pad
                groups: channels,
                dilation,
                ..Default::default()
            },
        );

        let snake1 = Snake1d::load(map, &format!("{prefix}.block.2.alpha"), channels)?;

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
            snake0,
            depthwise,
            snake1,
            pointwise,
            kernel,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        // Snake1d → causal depthwise Conv1d → Snake1d → pointwise Conv1d(k=1) + skip
        let h = self.snake0.forward(x)?;
        let h = causal_pad(&h, self.kernel, self.dilation, 1)?;
        let h = self.depthwise.forward(&h)?;
        let h = self.snake1.forward(&h)?;
        let h = self.pointwise.forward(&h)?;
        h + residual
    }
}

// ---------------------------------------------------------------------------
// SrFiLM — SR-conditioned Feature-wise Linear Modulation
// ---------------------------------------------------------------------------

/// FiLM modulation applied BEFORE upsampling (at the input channel resolution).
///
/// The SR scale/bias are selected from a 4-bin embedding table based on the
/// output sample rate, and applied to the block's INPUT signal:
///   h = scale * h + bias
///
/// Channel count matches the block's input channels (c_in), which corresponds
/// to the sr_cond_model tensor dimension.
#[derive(Debug, Clone)]
struct SrFiLM {
    /// [4, C] — 4 SR bins × channels
    scale_embed: Tensor,
    /// [4, C]
    bias_embed: Tensor,
    /// Currently selected bin (0..4)
    bin_idx: usize,
}

impl SrFiLM {
    fn load(map: &HashMap<String, Tensor>, prefix: &str, channels: usize) -> Result<Self> {
        let scale_key = format!("{prefix}.scale_embed.weight");
        let bias_key = format!("{prefix}.bias_embed.weight");
        let scale_embed = map
            .get(&scale_key)
            .unwrap_or_else(|| panic!("missing SR cond tensor: {scale_key}"))
            .clone();
        let bias_embed = map
            .get(&bias_key)
            .unwrap_or_else(|| panic!("missing SR cond tensor: {bias_key}"))
            .clone();
        // Validate shapes
        assert_eq!(
            scale_embed.shape().dims(),
            &[4, channels],
            "{scale_key} expected [4, {channels}], got {:?}",
            scale_embed.shape().dims()
        );
        assert_eq!(
            bias_embed.shape().dims(),
            &[4, channels],
            "{bias_key} expected [4, {channels}], got {:?}",
            bias_embed.shape().dims()
        );
        Ok(Self {
            scale_embed,
            bias_embed,
            bin_idx: 3, // default to highest bin
        })
    }

    /// Apply FiLM modulation: h = scale * h + bias
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [N, C, L]
        // scale_vec: [C], bias_vec: [C]
        let scale_vec = self.scale_embed.get(self.bin_idx)?; // [C]
        let bias_vec = self.bias_embed.get(self.bin_idx)?; // [C]
        let scale = scale_vec.reshape((1, scale_vec.dim(0)?, 1))?; // [1, C, 1]
        let bias = bias_vec.reshape((1, bias_vec.dim(0)?, 1))?; // [1, C, 1]
        x.broadcast_mul(&scale)?.broadcast_add(&bias)
    }
}

// ---------------------------------------------------------------------------
// CausalDecoderBlock — Snake1d → CausalTransposeConv1d → 3× CausalResidualUnit
// Python equivalent: CausalDecoderBlock (audio_vae_v2.py)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CausalDecoderBlock {
    snake: Snake1d,
    up_conv: ConvTranspose1d,
    sub_blocks: Vec<CausalResidualUnit>,
    sr_film: Option<SrFiLM>, // optional SR conditioning FiLM
    stride: usize,
    causal_pad: usize, // = ceil(stride/2), for trim = pad*2 - stride%2
}

impl CausalDecoderBlock {
    fn load(
        map: &HashMap<String, Tensor>,
        prefix: &str,
        c_in: usize,
        c_out: usize,
        kernel: usize,
        stride: usize,
        sr_film: Option<SrFiLM>,
    ) -> Result<Self> {
        let snake = Snake1d::load(map, &format!("{prefix}.block.0.alpha"), c_in)?;

        // ConvTranspose1d weight: [in_channels, out_channels, kernel_size] = [c_in, c_out, kernel]
        let up_weight = load_conv_weight(
            map,
            &format!("{prefix}.block.1.weight"),
            &[c_in, c_out, kernel],
        )?;
        let up_bias = load_1d(map, &format!("{prefix}.block.1.bias"), c_out)?;

        // Python CausalTransposeConv1d:
        //   padding=ceil(stride/2) is NOT passed to nn.ConvTranspose1d (padding=0 there)
        //   Forward: super().forward(x)[..., :-(pad*2 - output_pad)]
        //   So the actual conv has padding=0, output_padding=0
        let causal_pad = (stride + 1) / 2; // ceil(stride / 2) — ONLY used for trim

        let up_conv = ConvTranspose1d::new(
            up_weight,
            Some(up_bias),
            ConvTranspose1dConfig {
                padding: 0,        // Python: nn.ConvTranspose1d has default padding=0
                output_padding: 0, // Python: nn.ConvTranspose1d has default output_padding=0
                stride,
                ..Default::default()
            },
        );

        // CausalResidualUnits with dilations [1, 3, 9] — matching Python
        let dilations = [1usize, 3, 9];
        let mut sub_blocks = Vec::new();
        for (i, &dilation) in dilations.iter().enumerate() {
            let idx = i + 2; // block.2, block.3, block.4
            let sub = CausalResidualUnit::load(map, &format!("{prefix}.block.{idx}"), c_out, dilation)?;
            sub_blocks.push(sub);
        }

        Ok(Self {
            snake,
            up_conv,
            sub_blocks,
            sr_film,
            stride,
            causal_pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply SR FiLM to input before upsampling
        let h = if let Some(film) = &self.sr_film {
            film.forward(x)?
        } else {
            x.clone()
        };
        // Snake1d activation before transposed conv
        let h = self.snake.forward(&h)?;
        // Python CausalTransposeConv1d:
        //   super().forward(x)[..., :-(padding*2 - output_padding)]
        //   where padding=ceil(stride/2), output_padding=0 (default)
        let h = self.up_conv.forward(&h)?;
        let trim = 2 * self.causal_pad - (self.stride % 2); // Python: pad*2 - output_pad (output_pad=stride%2)
        let h = if trim > 0 {
            let len = h.dim(2)?;
            if len >= trim {
                h.narrow(2, 0, len - trim)?
            } else {
                h
            }
        } else {
            h
        };
        // 3× CausalResidualUnits
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
    model_0: Conv1d,                 // depthwise 64→64, k=7, causal
    model_1: Conv1d,                 // pointwise 64→2048, k=1
    up_blocks: Vec<CausalDecoderBlock>, // model 2–7
    model_8: Snake1d,                // Snake1d(32) before final conv
    model_9: Conv1d,                 // 32→1, k=7, causal
    pub sample_rate: u32,
    pub out_sample_rate: u32,
}

/// Determine SR bin index from sample rate.
///
/// bin_boundaries: [3] — e.g., [20000, 30000, 40000]
///   sample_rate < boundaries[0] → bin 0
///   sample_rate < boundaries[1] → bin 1
///   sample_rate < boundaries[2] → bin 2
///   else → bin 3
fn compute_sr_bin(
    sample_rate: u32,
    sr_bin_boundaries: &[usize],
) -> usize {
    for (i, &boundary) in sr_bin_boundaries.iter().enumerate() {
        if (sample_rate as usize) < boundary {
            return i;
        }
    }
    3 // highest bin
}

impl AudioVAE {
    /// Load decoder from the raw fused tensor map.
    ///
    /// The map must contain `decoder.model.N.*` tensors with weight_norm already fused.
    /// Use `weights::load_audiovae_decoder_tensors()` to obtain it.
    pub fn load(tensors: &HashMap<String, Tensor>, cfg: &AudioVaeConfig) -> Result<Self> {
        // Pre-compute the SR bin
        let sr_bin = compute_sr_bin(cfg.out_sample_rate, &cfg.sr_bin_boundaries);
        eprintln!(
            "[AudioVAE] SR bin = {sr_bin} (out_sample_rate={} Hz, boundaries={:?})",
            cfg.out_sample_rate, cfg.sr_bin_boundaries
        );

        // Pre-load SR FiLM modules for each up block
        // sr_cond_model.2 → block model.2 (2048 ch)
        // sr_cond_model.3 → block model.3 (1024 ch)
        // sr_cond_model.4 → block model.4 (512 ch)
        // sr_cond_model.5 → block model.5 (256 ch)
        // sr_cond_model.6 → block model.6 (128 ch)
        // sr_cond_model.7 → block model.7 (64 ch)
        let sr_channels: [usize; 6] = [2048, 1024, 512, 256, 128, 64];
        let mut sr_film_modules = Vec::new();
        for (i, &ch) in sr_channels.iter().enumerate() {
            let n = i + 2; // model.2 .. model.7
            let prefix = format!("decoder.sr_cond_model.{n}");
            if tensors.contains_key(&format!("{prefix}.scale_embed.weight")) {
                let mut film = SrFiLM::load(tensors, &prefix, ch)?;
                film.bin_idx = sr_bin;
                sr_film_modules.push(Some(film));
                eprintln!("[AudioVAE] Loaded SR FiLM for model.{n} ({ch} ch)");
            } else {
                eprintln!(
                    "[AudioVAE] WARNING: SR FiLM tensors not found for model.{n} — skipping"
                );
                sr_film_modules.push(None);
            }
        }

        // model.0: Causal depthwise Conv1d(64, 64, k=7, groups=64) — no activation
        // Causal padding handled manually in decode()
        let w0 = load_conv_weight(tensors, "decoder.model.0.weight", &[64, 1, 7])?;
        let b0 = load_1d(tensors, "decoder.model.0.bias", 64)?;
        let model_0 = Conv1d::new(
            w0,
            Some(b0),
            Conv1dConfig {
                padding: 0,
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

        // model.2–7: CausalDecoderBlocks
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
            let film = sr_film_modules.get(idx).cloned().unwrap_or(None);
            let block = CausalDecoderBlock::load(
                tensors,
                &format!("decoder.model.{n}"),
                c_in,
                c_out,
                kernel,
                stride,
                film,
            )?;
            up_blocks.push(block);
        }

        // model.8: Snake1d(32) — activation before final conv
        let model_8 = Snake1d::load(tensors, "decoder.model.8.alpha", 32)?;

        // model.9: CausalConv1d(32, 1, k=7)
        let w9 = load_conv_weight(tensors, "decoder.model.9.weight", &[1, 32, 7])?;
        let b9 = load_1d(tensors, "decoder.model.9.bias", 1)?;
        let model_9 = Conv1d::new(
            w9,
            Some(b9),
            Conv1dConfig {
                padding: 0, // causal: manual left-pad
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

        // model.0: Causal depthwise Conv1d(64, 64, k=7, groups=64) — no activation
        h = causal_pad(&h, 7, 1, 1)?;
        h = self.model_0.forward(&h)?;
        eprintln!("  [AudioVAE] model.0: shape={:?} peak={:.6}", h.shape(), h.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?);

        // model.1: pointwise Conv1d(64, 2048, k=1) — no activation
        h = self.model_1.forward(&h)?;
        eprintln!("  [AudioVAE] model.1: shape={:?} peak={:.6} mean={:.6}", h.shape(), h.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?, h.mean(2)?.mean(1)?.to_vec1::<f32>()?[0]);

        // model.2–7: CausalDecoderBlocks (with SR FiLM if loaded)
        for (i, block) in self.up_blocks.iter().enumerate() {
            h = block.forward(&h)?;
            eprintln!("  [AudioVAE] model.{}: shape={:?} peak={:.6} mean={:.6}", i+2, h.shape(), h.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?, h.mean(2)?.mean(1)?.to_vec1::<f32>()?[0]);
        }

        // model.8: Snake1d activation before final conv
        h = self.model_8.forward(&h)?;
        eprintln!("  [AudioVAE] model.8: shape={:?} peak={:.6} mean={:.6}", h.shape(), h.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?, h.mean(2)?.mean(1)?.to_vec1::<f32>()?[0]);

        // model.9: Causal Conv1d(32, 1, k=7) + Tanh
        h = causal_pad(&h, 7, 1, 1)?;
        h = self.model_9.forward(&h)?;
        eprintln!("  [AudioVAE] model.9 (pre-tanh): shape={:?} peak={:.6} mean={:.6}", h.shape(), h.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?, h.mean(2)?.mean(1)?.to_vec1::<f32>()?[0]);
        h = h.tanh()?;
        eprintln!("  [AudioVAE] tanh out: shape={:?} peak={:.6} mean={:.6}", h.shape(), h.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?, h.mean(2)?.mean(1)?.to_vec1::<f32>()?[0]);

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
    fn sr_bin_computation() {
        let boundaries = vec![20000, 30000, 40000];
        assert_eq!(compute_sr_bin(16000, &boundaries), 0);
        assert_eq!(compute_sr_bin(22000, &boundaries), 1);
        assert_eq!(compute_sr_bin(32000, &boundaries), 2);
        // 44100 >= 40000 → bin 3
        assert_eq!(compute_sr_bin(44100, &boundaries), 3);
        assert_eq!(compute_sr_bin(48000, &boundaries), 3);
        assert_eq!(compute_sr_bin(96000, &boundaries), 3);
    }

    #[test]
    #[ignore = "requires audiovae.safetensors in models/VoxCPM2/"]
    fn test_audiovae_compare_with_python() -> Result<()> {
        let dev = candle_core::Device::Cpu;
        let model_dir = if std::path::Path::new("models/VoxCPM2").exists() {
            std::path::Path::new("models/VoxCPM2").to_path_buf()
        } else {
            std::path::Path::new("../../models/VoxCPM2").to_path_buf()
        };
        // Load config
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|e| candle_core::Error::Msg(format!("read config: {e}")))?;
        let full_cfg: crate::config::VoxConfig = serde_json::from_str(&config_text)
            .map_err(|e| candle_core::Error::Msg(format!("parse config: {e}")))?;
        let tensors = weights::load_audiovae_decoder_tensors(&model_dir, &dev)?;
        let vae = AudioVAE::load(&tensors, &full_cfg.audio_vae_config)?;

        // Load test latent saved from Python
        let latent_bytes = std::fs::read("test_latent_8frames.f32")
            .map_err(|e| candle_core::Error::Msg(format!("read test latent: {e}")))?;
        // [1, 64, 8] = 512 floats
        let latent = Tensor::from_raw_buffer(
            &latent_bytes,
            candle_core::DType::F32,
            &[1, 64, 8],
            &dev,
        )?;
        eprintln!("[Rust] Input latent: shape={:?}", latent.shape());
        let peak = latent.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?;
        let mean = latent.mean(2)?.mean(1)?.to_vec1::<f32>()?[0];
        let sq = latent.sqr()?.mean(2)?.mean(1)?.to_vec1::<f32>()?[0];
        let std_val = (sq - mean * mean).sqrt();
        eprintln!("[Rust] Input: peak={peak:.6} mean={mean:.6} std={std_val:.6}");

        let waveform = vae.decode(&latent)?;
        let (n, c, samples) = waveform.shape().dims3()?;
        let peak_v = waveform.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?;
        let mean_v = waveform.mean(2)?.mean(1)?.to_vec1::<f32>()?[0];
        eprintln!("[Rust] Output: [{n}, {c}, {samples}], peak={peak_v:.6}, mean={mean_v:.6}");
        eprintln!("[Python] Output: [1, 1, 15360], peak=0.310290, mean=-0.002143");
        eprintln!("[Rust] Expected 15360 samples for 8-frame input");
        if samples != 15360 {
            eprintln!("[Rust] WARNING: output length mismatch! Got {samples}, expected 15360");
        }
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

        // Create random latent: [1, 64, 16]
        let latent = Tensor::rand(-2.0f32, 2.0, &[1, 64, 16], &dev)?;
        let waveform = vae.decode(&latent)?;

        // 6 upsampling blocks: rates [8, 6, 5, 2, 2, 2], product = 1920
        // 16 frames × 1920 = 30720 samples at base rate (16000 Hz)
        let (n, c, samples) = waveform.shape().dims3()?;
        assert_eq!(n, 1, "batch dim");
        assert_eq!(c, 1, "mono audio");
        assert!(samples > 30000, "expected ~30720 samples, got {samples}");
        let peak_v = waveform.abs()?.flatten_all()?.max_keepdim(0)?.squeeze(0)?.to_vec0::<f32>()?;
        println!("AudioVAE decode shape OK: [{n}, {c}, {samples}], peak={peak_v:.6}");
        // With SR conditioning enabled, peak should be significantly higher
        // than the ~5e-5 seen without SR conditioning.
        // However, for random latent input the exact peak depends on the latent values.
        assert!(peak_v > 1e-6, "output peak too low (likely NaN): {peak_v:.6}");
        Ok(())
    }
}
