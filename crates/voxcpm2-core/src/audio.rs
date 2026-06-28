use std::path::Path;

const SPEECH_LOWPASS_HZ: f32 = 12_000.0;
const FADE_IN_MS: f32 = 5.0;
const FADE_OUT_MS: f32 = 12.0;
const PCM_HEADROOM: f32 = 0.95;

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioPolishReport {
    pub dc_offset: f32,
    pub peak_before: f32,
    pub peak_after: f32,
    pub headroom_gain: f32,
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
    let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
    for sample in samples.iter_mut() {
        *sample -= dc_offset;
    }

    apply_one_pole_lowpass_zero_phase(samples, sample_rate, SPEECH_LOWPASS_HZ);
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

fn lowpass_pass(samples: &mut [f32], alpha: f32) {
    let mut y = samples[0];
    for sample in samples.iter_mut().skip(1) {
        y += alpha * (*sample - y);
        *sample = y;
    }
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

    fn sine(len: usize, sample_rate: u32, hz: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * hz * t).sin()
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }
}
