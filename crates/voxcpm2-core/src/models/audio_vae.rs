//! AudioVAE V2 Decoder — 將潛變量 (latent) 解碼為 48kHz 音訊。
//!
//! 對應 tensor name prefix: `encoder.*` / `decoder.*`
//!
//! 注意: 從 safetensors 載入時，由於原始 audiovae.pth 使用 weight normalization
//! (weight_g/weight_v)，我們需要特殊處理。

use crate::config::AudioVaeConfig;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

/// 單一卷積區塊（1D convolution + 可選上採樣 + 激活）。
pub struct ConvBlock {
    conv: Conv1d,
    upsample: Option<usize>,
    activation: candle_nn::Activation,
}

impl ConvBlock {
    pub fn new(conv: Conv1d, upsample: Option<usize>) -> Self {
        Self {
            conv,
            upsample,
            activation: candle_nn::Activation::Silu,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        let h = self.activation.forward(&h)?;
        if let Some(rate) = self.upsample {
            // 最近鄰插值上採樣
            let (batch, channels, length) = h.shape().dims3()?;
            let new_len = length * rate;
            h.unsqueeze(3)?
                .expand((batch, channels, length, rate))?
                .reshape((batch, channels, new_len))
        } else {
            Ok(h)
        }
    }
}

/// AudioVAE V2 Decoder。
pub struct AudioVAE {
    pub decoder_convs: Vec<ConvBlock>,
    pub final_conv: Conv1d,
    pub sr_convs: Vec<ConvBlock>,
    pub sample_rate: u32,
    pub out_sample_rate: u32,
}

impl AudioVAE {
    /// 從 VarBuilder 載入 AudioVAE decoder。
    ///
    /// 注意：原始的 `.pth` 使用 weight normalization (weight_g + weight_v)，
    /// 但轉換後的 safetensors 已經合併為標準 weight 張量，所以可以直接載入。
    pub fn load(vb: &VarBuilder, cfg: &AudioVaeConfig) -> Result<Self> {
        let _dev = vb.device();
        let _ = DType::F32;

        // Decoder 路徑
        let mut decoder_convs = Vec::new();
        let mut in_ch = cfg.latent_dim;
        for (i, &rate) in cfg.decoder_rates.iter().enumerate() {
            let out_ch = if i == cfg.decoder_rates.len() - 1 {
                // 最後一層輸出到 1 channel（mono audio）
                1
            } else {
                cfg.decoder_dim.min(in_ch * rate)
            };
            let conv_cfg = Conv1dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            };
            let conv = candle_nn::conv1d_no_bias(
                in_ch,
                out_ch,
                3,
                conv_cfg,
                vb.pp(format!("decoder.block.{i}")),
            )?;
            decoder_convs.push(ConvBlock::new(conv, Some(rate)));
            in_ch = out_ch;
        }

        // Final conv (1x1) to get proper audio output
        let final_conv_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        };
        let final_conv = candle_nn::conv1d_no_bias(1, 1, 1, final_conv_cfg, vb.pp("decoder.out"))?;

        // Super-resolution path（sr_bin_boundaries 定義分段頻率邊界）
        let mut sr_convs = Vec::new();
        let mut sr_in = 1;
        for (i, &rate) in cfg.sr_bin_boundaries.iter().enumerate() {
            let out_ch = sr_in;
            let conv_cfg = Conv1dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            };
            let conv = candle_nn::conv1d_no_bias(
                sr_in,
                out_ch,
                3,
                conv_cfg,
                vb.pp(format!("decoder.sr_block.{i}")),
            )?;
            sr_convs.push(ConvBlock::new(conv, Some(rate / sr_in)));
            sr_in = out_ch;
        }

        Ok(Self {
            decoder_convs,
            final_conv,
            sr_convs,
            sample_rate: cfg.sample_rate,
            out_sample_rate: cfg.out_sample_rate,
        })
    }

    /// 解碼 latent 到 48kHz 波形。
    /// latent: [batch, latent_dim, time]
    /// 回傳: [batch, 1, samples] (mono audio)
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let mut h = latent.clone();

        // Decoder blocks with upsampling
        for block in &self.decoder_convs {
            h = block.forward(&h)?;
        }

        // Final conv
        h = self.final_conv.forward(&h)?;

        // Super-resolution path（若存在）
        for block in &self.sr_convs {
            h = block.forward(&h)?;
        }

        Ok(h)
    }

    /// 對音訊進行 sanity check（peak, NaN, duration）。
    pub fn check_audio(&self, waveform: &Tensor) -> Result<()> {
        let vals = waveform.flatten_all()?.to_vec1::<f32>()?;
        super::super::audio::check_audio(&vals)
            .map_err(|e| candle_core::Error::Msg(format!("AudioVAE check: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audiovae_smoke() -> Result<()> {
        let _cfg = AudioVaeConfig {
            encoder_dim: 128,
            encoder_rates: vec![2, 5, 8, 8],
            latent_dim: 64,
            decoder_dim: 64,
            decoder_rates: vec![8, 6, 5, 2, 2, 2],
            sr_bin_boundaries: vec![20000, 30000, 40000],
            sample_rate: 16000,
            out_sample_rate: 48000,
        };

        // Smoke test: verify config constructs
        Ok(())
    }
}
