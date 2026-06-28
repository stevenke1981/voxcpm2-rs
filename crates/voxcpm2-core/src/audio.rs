use std::path::Path;

/// Lowpass cutoff for hiss reduction (12kHz, preserves full speech band).
const SPEECH_LOWPASS_HZ: f32 = 12_000.0;
/// Highpass cutoff for sub-bass rumble from imperfect CFM latents.
const SPEECH_HIGHPASS_HZ: f32 = 80.0;
const EXPANDER_THRESHOLD: f32 = 0.006;
const EXPANDER_FLOOR_GAIN: f32 = 0.30;
const EXPANDER_ATTACK_MS: f32 = 8.0;
const EXPANDER_RELEASE_MS: f32 = 120.0;

const FADE_IN_MS: f32 = 5.0;
const FADE_OUT_MS: f32 = 12.0;
const PCM_HEADROOM: f32 = 0.95;

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioPolishReport {
    pub dc_offset: f32,
    pub peak_before: f32,
    pub peak_after: f32,
    pub headroom_gain: f32,
    pub quiet_rms_before: f32,
    pub quiet_rms_after: f32,
}

/// Load a mono WAV file and return (samples, sample_rate).
/// Supports 16-bit PCM (most common). Other formats may fail.
pub fn load_wav_mono(path: impl AsRef<Path>) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path.as_ref()).map_err(|e| anyhow::anyhow!("open WAV: {e}"))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match (spec.channels, spec.bits_per_sample, spec.sample_format) {
        (1, 16, hound::SampleFormat::Int) => reader
            .samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / i16::MAX as f32)
            .collect(),
        (2, 16, hound::SampleFormat::Int) => {
            // Downmix stereo to mono
            let stereo: Vec<f32> = reader
                .samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / i16::MAX as f32)
                .collect();
            stereo.chunks(2).map(|ch| (ch[0] + ch[1]) * 0.5).collect()
        }
        _ => anyhow::bail!(
            "unsupported WAV format: {}ch {}bit {:?}",
            spec.channels,
            spec.bits_per_sample,
            spec.sample_format,
        ),
    };

    Ok((samples, sample_rate))
}

/// Simple linear interpolation resampling.
/// `src_rate` → `dst_rate`. Only downsamples (src_rate >= dst_rate).
pub fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;
        if src_idx + 1 < samples.len() {
            let a = samples[src_idx] as f64;
            let b = samples[src_idx + 1] as f64;
            out.push((a + (b - a) * frac) as f32);
        } else {
            out.push(samples.last().copied().unwrap_or(0.0));
        }
    }
    out
}

pub fn write_wav_f32(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> anyhow::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn polish_generated_speech(samples: &mut [f32], sample_rate: u32) -> AudioPolishReport {
    if samples.is_empty() {
        return AudioPolishReport::default();
    }

    let peak_before = peak(samples);
    let quiet_rms_before = quiet_rms(samples, sample_rate);
    let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
    for sample in samples.iter_mut() {
        *sample -= dc_offset;
    }

    // Remove residual CFM/AudiVAE rumble and hiss while preserving speech.
    apply_one_pole_highpass_zero_phase(samples, sample_rate, SPEECH_HIGHPASS_HZ);
    apply_one_pole_lowpass_zero_phase(samples, sample_rate, SPEECH_LOWPASS_HZ);
    apply_soft_expander(samples, sample_rate);
    apply_edge_fades(samples, sample_rate, FADE_IN_MS, FADE_OUT_MS);

    let peak_after_filter = peak(samples);
    let headroom_gain = if peak_after_filter > PCM_HEADROOM {
        PCM_HEADROOM / peak_after_filter
    } else {
        1.0
    };
    if headroom_gain < 1.0 {
        for sample in samples.iter_mut() {
            *sample *= headroom_gain;
        }
    }

    AudioPolishReport {
        dc_offset,
        peak_before,
        peak_after: peak(samples),
        headroom_gain,
        quiet_rms_before,
        quiet_rms_after: quiet_rms(samples, sample_rate),
    }
}

pub fn smoke_tone(text: &str, sample_rate: u32) -> Vec<f32> {
    let seconds = (0.35 + (text.chars().count() as f32 * 0.018)).clamp(0.35, 3.0);
    let n = (seconds * sample_rate as f32) as usize;
    let freq = 220.0 + (text.chars().count() as f32 % 12.0) * 20.0;
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let env = (1.0 - (i as f32 / n as f32)).clamp(0.0, 1.0);
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.15 * env
        })
        .collect()
}

pub fn check_audio(samples: &[f32]) -> anyhow::Result<()> {
    if samples.is_empty() {
        anyhow::bail!("audio buffer is empty");
    }
    if samples.iter().any(|x| !x.is_finite()) {
        anyhow::bail!("audio contains NaN/Inf");
    }
    let peak = samples.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
    if peak > 1.5 {
        anyhow::bail!("audio peak too high: {peak}");
    }
    Ok(())
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |acc, x| acc.max(x.abs()))
}

fn apply_one_pole_lowpass_zero_phase(samples: &mut [f32], sample_rate: u32, cutoff_hz: f32) {
    if samples.len() < 2 || sample_rate == 0 || cutoff_hz <= 0.0 {
        return;
    }
    let nyquist = sample_rate as f32 * 0.5;
    if cutoff_hz >= nyquist {
        return;
    }

    let alpha = 1.0 - (-2.0 * std::f32::consts::PI * cutoff_hz / sample_rate as f32).exp();
    lowpass_pass(samples, alpha);
    samples.reverse();
    lowpass_pass(samples, alpha);
    samples.reverse();
}

fn apply_one_pole_highpass_zero_phase(samples: &mut [f32], sample_rate: u32, cutoff_hz: f32) {
    if samples.len() < 2 || sample_rate == 0 || cutoff_hz <= 0.0 {
        return;
    }
    let mut low = samples.to_vec();
    apply_one_pole_lowpass_zero_phase(&mut low, sample_rate, cutoff_hz);
    for (sample, low_sample) in samples.iter_mut().zip(low) {
        *sample -= low_sample;
    }
}

fn lowpass_pass(samples: &mut [f32], alpha: f32) {
    let mut y = samples[0];
    for sample in samples.iter_mut().skip(1) {
        y += alpha * (*sample - y);
        *sample = y;
    }
}

fn apply_soft_expander(samples: &mut [f32], sample_rate: u32) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }
    let attack = smoothing_coeff(EXPANDER_ATTACK_MS, sample_rate);
    let release = smoothing_coeff(EXPANDER_RELEASE_MS, sample_rate);
    let mut env = 0.0f32;
    for sample in samples.iter_mut() {
        let level = sample.abs();
        let coeff = if level > env { attack } else { release };
        env = coeff.mul_add(env, (1.0 - coeff) * level);
        let gain = if env >= EXPANDER_THRESHOLD {
            1.0
        } else {
            let openness = (env / EXPANDER_THRESHOLD).clamp(0.0, 1.0);
            EXPANDER_FLOOR_GAIN + (1.0 - EXPANDER_FLOOR_GAIN) * openness
        };
        *sample *= gain;
    }
}

fn smoothing_coeff(ms: f32, sample_rate: u32) -> f32 {
    let samples = (ms.max(0.1) * sample_rate as f32 / 1000.0).max(1.0);
    (-1.0 / samples).exp()
}

fn quiet_rms(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }
    let frame = (sample_rate as usize / 50).max(1);
    let frames = samples.len() / frame;
    if frames == 0 {
        return rms(samples);
    }
    let mut rms_values: Vec<f32> = (0..frames)
        .map(|i| rms(&samples[i * frame..(i + 1) * frame]))
        .collect();
    rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    rms_values[(rms_values.len() / 10).min(rms_values.len() - 1)]
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn apply_edge_fades(samples: &mut [f32], sample_rate: u32, fade_in_ms: f32, fade_out_ms: f32) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }

    let fade_in = ((sample_rate as f32 * fade_in_ms) / 1000.0).round() as usize;
    apply_fade_in(samples, fade_in.min(samples.len()));

    let fade_out = ((sample_rate as f32 * fade_out_ms) / 1000.0).round() as usize;
    apply_fade_out(samples, fade_out.min(samples.len()));
}

fn apply_fade_in(samples: &mut [f32], len: usize) {
    if len < 2 {
        return;
    }
    for (i, sample) in samples.iter_mut().take(len).enumerate() {
        let phase = i as f32 / (len - 1) as f32;
        let gain = 0.5 - 0.5 * (std::f32::consts::PI * phase).cos();
        *sample *= gain;
    }
}

fn apply_fade_out(samples: &mut [f32], len: usize) {
    if len < 2 {
        return;
    }
    let start = samples.len() - len;
    for (i, sample) in samples.iter_mut().skip(start).enumerate() {
        let phase = i as f32 / (len - 1) as f32;
        let gain = 0.5 + 0.5 * (std::f32::consts::PI * phase).cos();
        *sample *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_tone_is_valid() {
        let s = smoke_tone("hello", 48_000);
        check_audio(&s).unwrap();
        assert!(s.len() > 10_000);
    }

    #[test]
    fn polish_removes_dc_and_keeps_headroom() {
        let mut samples = vec![1.05, 0.55, -0.45, 0.25, -1.1, 0.35, -0.2, 0.1];
        let report = polish_generated_speech(&mut samples, 48_000);
        assert!(report.dc_offset.abs() > 0.01);
        assert!(peak(&samples) <= PCM_HEADROOM + 1e-6);
        assert!(samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn lowpass_attenuates_ultrasonic_hiss_more_than_voice_band() {
        let sample_rate = 48_000;
        let len = sample_rate as usize / 5;
        let mut voice = sine(len, sample_rate, 1_000.0);
        let mut hiss = sine(len, sample_rate, 18_000.0);
        let voice_before = rms(&voice);
        let hiss_before = rms(&hiss);

        apply_one_pole_lowpass_zero_phase(&mut voice, sample_rate, SPEECH_LOWPASS_HZ);
        apply_one_pole_lowpass_zero_phase(&mut hiss, sample_rate, SPEECH_LOWPASS_HZ);

        let voice_ratio = rms(&voice) / voice_before;
        let hiss_ratio = rms(&hiss) / hiss_before;
        assert!(voice_ratio > 0.95, "voice band should be mostly preserved");
        assert!(hiss_ratio < 0.60, "18kHz hiss should be attenuated");
    }

    #[test]
    fn highpass_reduces_rumble_more_than_voice_band() {
        let sample_rate = 48_000;
        let len = sample_rate as usize / 5;
        let mut voice = sine(len, sample_rate, 1_000.0);
        let mut rumble = sine(len, sample_rate, 30.0);
        let voice_before = rms(&voice);
        let rumble_before = rms(&rumble);

        apply_one_pole_highpass_zero_phase(&mut voice, sample_rate, SPEECH_HIGHPASS_HZ);
        apply_one_pole_highpass_zero_phase(&mut rumble, sample_rate, SPEECH_HIGHPASS_HZ);

        let voice_ratio = rms(&voice) / voice_before;
        let rumble_ratio = rms(&rumble) / rumble_before;
        assert!(voice_ratio > 0.95, "voice band should be mostly preserved");
        assert!(rumble_ratio < 0.35, "30Hz rumble should be attenuated");
    }

    #[test]
    fn soft_expander_reduces_quiet_noise_and_preserves_loud_speech() {
        let sample_rate = 48_000;
        let mut samples = vec![0.002; sample_rate as usize / 10];
        samples.extend(vec![0.04; sample_rate as usize / 10]);

        let quiet_before = rms(&samples[..sample_rate as usize / 10]);
        let loud_before = rms(&samples[sample_rate as usize / 10..]);
        apply_soft_expander(&mut samples, sample_rate);
        let quiet_after = rms(&samples[..sample_rate as usize / 10]);
        let loud_after = rms(&samples[sample_rate as usize / 10..]);

        assert!(quiet_after < quiet_before * 0.75);
        assert!(loud_after > loud_before * 0.90);
    }

    fn sine(len: usize, sample_rate: u32, hz: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * hz * t).sin()
            })
            .collect()
    }
}
