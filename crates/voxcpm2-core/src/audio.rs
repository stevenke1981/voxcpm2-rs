use serde::{Deserialize, Serialize};
use std::path::Path;

/// Lowpass cutoff for hiss reduction (12kHz, preserves full speech band).
const SPEECH_LOWPASS_HZ: f32 = 12_000.0;
/// Highpass cutoff for sub-bass rumble from imperfect CFM latents.
const SPEECH_HIGHPASS_HZ: f32 = 80.0;
const EXPANDER_THRESHOLD: f32 = 0.006;
const EXPANDER_FLOOR_GAIN: f32 = 0.30;
const EXPANDER_ATTACK_MS: f32 = 8.0;
const EXPANDER_RELEASE_MS: f32 = 120.0;
const BACKGROUND_GATE_FRAME_MS: f32 = 20.0;
const BACKGROUND_GATE_FLOOR_GAIN: f32 = 0.12;
const CLONE_REF_GATE_FLOOR_GAIN: f32 = 0.18;
const HARSH_SMOOTHER_FRAME_MS: f32 = 20.0;
const HARSH_SMOOTHER_LOWPASS_HZ: f32 = 4_500.0;
const HARSH_ZCR_THRESHOLD: f32 = 0.11;
const HARSH_RMS_THRESHOLD: f32 = 0.005;
const HARSH_MAX_BLEND: f32 = 0.70;
const PATCH_BOUNDARY_SMOOTHER_PATCH_MS: f32 = 160.0;
const PATCH_BOUNDARY_SMOOTHER_WINDOW_MS: f32 = 20.0;
const PATCH_BOUNDARY_SMOOTHER_LOWPASS_HZ: f32 = 4_000.0;
const PATCH_BOUNDARY_MAX_BLEND: f32 = 0.85;
const HIGHBAND_LEVELER_FRAME_MS: f32 = 20.0;
const HIGHBAND_LEVELER_SPLIT_HZ: f32 = 4_000.0;
const HIGHBAND_LEVELER_MAX_RMS: f32 = 0.0075;
const HIGHBAND_LEVELER_RATIO: f32 = 0.12;
const HIGHBAND_LEVELER_FLOOR: f32 = 0.0012;
const HIGHBAND_LEVELER_MIN_GAIN: f32 = 0.20;

const FADE_IN_MS: f32 = 5.0;
const FADE_OUT_MS: f32 = 12.0;
const PCM_HEADROOM: f32 = 0.95;
pub const MAX_CLONE_REFERENCE_SECS: f64 = 30.0;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct AudioPolishReport {
    pub dc_offset: f32,
    pub peak_before: f32,
    pub peak_after: f32,
    pub headroom_gain: f32,
    pub quiet_rms_before: f32,
    pub quiet_rms_after: f32,
    pub background_gate_threshold: f32,
    pub harsh_frames_smoothed: usize,
    pub patch_boundaries_smoothed: usize,
    pub highband_frames_leveled: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CloneReferenceTrimReport {
    pub original_samples: usize,
    pub trimmed_samples: usize,
    pub sample_rate: u32,
    pub max_duration_sec: f64,
    pub original_duration_sec: f64,
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
    let background_gate_threshold = apply_adaptive_background_gate(
        samples,
        sample_rate,
        BACKGROUND_GATE_FLOOR_GAIN,
        1.4,
        0.0035,
        0.020,
    );
    let harsh_frames_smoothed = apply_harsh_midband_smoother(samples, sample_rate);
    let patch_boundaries_smoothed = apply_patch_boundary_smoother(samples, sample_rate);
    let highband_frames_leveled = apply_highband_residual_leveler(samples, sample_rate);
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
        background_gate_threshold,
        harsh_frames_smoothed,
        patch_boundaries_smoothed,
        highband_frames_leveled,
    }
}

pub fn polish_clone_reference_audio(samples: &mut [f32], sample_rate: u32) -> AudioPolishReport {
    if samples.is_empty() {
        return AudioPolishReport::default();
    }

    let peak_before = peak(samples);
    let quiet_rms_before = quiet_rms(samples, sample_rate);
    let dc_offset = samples.iter().sum::<f32>() / samples.len() as f32;
    for sample in samples.iter_mut() {
        *sample -= dc_offset;
    }

    apply_one_pole_highpass_zero_phase(samples, sample_rate, SPEECH_HIGHPASS_HZ);
    apply_one_pole_lowpass_zero_phase(
        samples,
        sample_rate,
        SPEECH_LOWPASS_HZ.min(sample_rate as f32 * 0.45),
    );
    let background_gate_threshold = apply_adaptive_background_gate(
        samples,
        sample_rate,
        CLONE_REF_GATE_FLOOR_GAIN,
        1.35,
        0.004,
        0.024,
    );
    let harsh_frames_smoothed = apply_harsh_midband_smoother(samples, sample_rate);
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
        background_gate_threshold,
        harsh_frames_smoothed,
        patch_boundaries_smoothed: 0,
        highband_frames_leveled: 0,
    }
}

pub fn trim_clone_reference_audio(
    samples: &mut Vec<f32>,
    sample_rate: u32,
) -> Option<CloneReferenceTrimReport> {
    if sample_rate == 0 {
        return None;
    }
    let max_samples = (MAX_CLONE_REFERENCE_SECS * sample_rate as f64) as usize;
    if samples.len() <= max_samples {
        return None;
    }

    let report = CloneReferenceTrimReport {
        original_samples: samples.len(),
        trimmed_samples: max_samples,
        sample_rate,
        max_duration_sec: MAX_CLONE_REFERENCE_SECS,
        original_duration_sec: samples.len() as f64 / sample_rate as f64,
    };
    samples.truncate(max_samples);
    Some(report)
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

fn apply_adaptive_background_gate(
    samples: &mut [f32],
    sample_rate: u32,
    floor_gain: f32,
    threshold_multiplier: f32,
    min_threshold: f32,
    max_threshold: f32,
) -> f32 {
    if samples.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    let frame_len = ((sample_rate as f32 * BACKGROUND_GATE_FRAME_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    let frame_count = samples.len().div_ceil(frame_len);
    if frame_count == 0 {
        return 0.0;
    }

    let mut frame_rms = Vec::with_capacity(frame_count);
    for frame_idx in 0..frame_count {
        let start = frame_idx * frame_len;
        let end = (start + frame_len).min(samples.len());
        frame_rms.push(rms(&samples[start..end]));
    }

    let mut sorted = frame_rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor_idx = (sorted.len() / 10).min(sorted.len() - 1);
    let noise_floor = sorted[floor_idx];
    let threshold = (noise_floor * threshold_multiplier).clamp(min_threshold, max_threshold);
    let full_open = (threshold * 2.2).max(threshold + 1e-6);
    let floor_gain = floor_gain.clamp(0.0, 1.0);

    let mut previous_gain = 1.0f32;
    for (frame_idx, &frame_level) in frame_rms.iter().enumerate() {
        let target_gain = if frame_level <= threshold {
            floor_gain
        } else if frame_level >= full_open {
            1.0
        } else {
            let openness = (frame_level - threshold) / (full_open - threshold);
            floor_gain + (1.0 - floor_gain) * openness.clamp(0.0, 1.0)
        };

        let start = frame_idx * frame_len;
        let end = (start + frame_len).min(samples.len());
        let span = (end - start).max(1) as f32;
        for (i, sample) in samples[start..end].iter_mut().enumerate() {
            let t = i as f32 / span;
            let gain = previous_gain + (target_gain - previous_gain) * t;
            *sample *= gain;
        }
        previous_gain = target_gain;
    }

    threshold
}

fn apply_harsh_midband_smoother(samples: &mut [f32], sample_rate: u32) -> usize {
    if samples.is_empty() || sample_rate == 0 {
        return 0;
    }

    let mut softened = samples.to_vec();
    apply_one_pole_lowpass_zero_phase(&mut softened, sample_rate, HARSH_SMOOTHER_LOWPASS_HZ);

    let frame_len = ((sample_rate as f32 * HARSH_SMOOTHER_FRAME_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    let frame_count = samples.len().div_ceil(frame_len);
    let mut previous_blend = 0.0f32;
    let mut smoothed_frames = 0usize;

    for frame_idx in 0..frame_count {
        let start = frame_idx * frame_len;
        let end = (start + frame_len).min(samples.len());
        let frame = &samples[start..end];
        let frame_rms = rms(frame);
        let zcr = zero_crossing_rate(frame);
        let zcr_open = ((zcr - HARSH_ZCR_THRESHOLD) / 0.08).clamp(0.0, 1.0);
        let rms_open = ((frame_rms - HARSH_RMS_THRESHOLD) / 0.015).clamp(0.0, 1.0);
        let target_blend = (zcr_open * rms_open * HARSH_MAX_BLEND).clamp(0.0, HARSH_MAX_BLEND);

        if target_blend > 0.01 {
            smoothed_frames += 1;
        }

        let span = (end - start).max(1) as f32;
        for i in 0..(end - start) {
            let t = i as f32 / span;
            let blend = previous_blend + (target_blend - previous_blend) * t;
            let idx = start + i;
            samples[idx] = samples[idx] * (1.0 - blend) + softened[idx] * blend;
        }
        previous_blend = target_blend;
    }

    smoothed_frames
}

fn apply_patch_boundary_smoother(samples: &mut [f32], sample_rate: u32) -> usize {
    if samples.is_empty() || sample_rate == 0 {
        return 0;
    }

    let patch_len = ((sample_rate as f32 * PATCH_BOUNDARY_SMOOTHER_PATCH_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    let window = ((sample_rate as f32 * PATCH_BOUNDARY_SMOOTHER_WINDOW_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    if samples.len() < patch_len + window * 2 {
        return 0;
    }

    let mut softened = samples.to_vec();
    apply_one_pole_lowpass_zero_phase(
        &mut softened,
        sample_rate,
        PATCH_BOUNDARY_SMOOTHER_LOWPASS_HZ,
    );

    let mut high_band = samples.to_vec();
    apply_one_pole_highpass_zero_phase(
        &mut high_band,
        sample_rate,
        PATCH_BOUNDARY_SMOOTHER_LOWPASS_HZ,
    );

    let mut smoothed = 0usize;
    let mut boundary = patch_len;
    while boundary + window < samples.len() {
        if boundary < window {
            boundary += patch_len;
            continue;
        }

        let left = boundary - window;
        let right = boundary + window;
        let before = &samples[left..boundary];
        let after = &samples[boundary..right];
        let rms_before = rms(before).max(1e-7);
        let rms_after = rms(after).max(1e-7);
        let rms_ratio = rms_after / rms_before;
        let high_before = rms(&high_band[left..boundary]).max(1e-7);
        let high_after = rms(&high_band[boundary..right]).max(1e-7);
        let high_ratio = high_after / high_before;
        let zcr_delta = (zero_crossing_rate(after) - zero_crossing_rate(before)).abs();
        let sample_step = (samples[boundary] - samples[boundary - 1]).abs();

        let should_smooth = !(0.50..=2.00).contains(&high_ratio)
            || !(0.45..=2.25).contains(&rms_ratio)
            || zcr_delta > 0.055
            || sample_step > 0.020;

        if should_smooth {
            smoothed += 1;
            let before_high_gain = if high_before > high_after * 1.60 {
                ((high_after * 1.40) / high_before).clamp(0.12, 0.70)
            } else {
                1.0
            };
            let after_high_gain = if high_after > high_before * 1.60 {
                ((high_before * 1.40) / high_after).clamp(0.12, 0.70)
            } else {
                1.0
            };
            for idx in left..right {
                let (distance, side_gain) = if idx < boundary {
                    ((boundary - idx) as f32 / window as f32, before_high_gain)
                } else {
                    ((idx - boundary) as f32 / window as f32, after_high_gain)
                };
                let distance = distance.clamp(0.0, 1.0);
                let envelope = 1.0 - distance;
                let base_high_gain = 1.0 - PATCH_BOUNDARY_MAX_BLEND * envelope * envelope;
                let leveled_high_gain = side_gain + (1.0 - side_gain) * distance.powi(4);
                let high_gain = base_high_gain.min(leveled_high_gain);
                samples[idx] = softened[idx] + (samples[idx] - softened[idx]) * high_gain;
            }
        }

        boundary += patch_len;
    }

    smoothed
}

fn apply_highband_residual_leveler(samples: &mut [f32], sample_rate: u32) -> usize {
    if samples.is_empty() || sample_rate == 0 {
        return 0;
    }

    let frame_len = ((sample_rate as f32 * HIGHBAND_LEVELER_FRAME_MS) / 1000.0)
        .round()
        .max(1.0) as usize;
    if samples.len() < frame_len {
        return 0;
    }

    let mut low = samples.to_vec();
    apply_one_pole_lowpass_zero_phase(&mut low, sample_rate, HIGHBAND_LEVELER_SPLIT_HZ);
    let high: Vec<f32> = samples
        .iter()
        .zip(&low)
        .map(|(sample, low)| sample - low)
        .collect();

    let frame_count = samples.len().div_ceil(frame_len);
    let mut target_gains = Vec::with_capacity(frame_count);
    let mut leveled_frames = 0usize;

    for frame_idx in 0..frame_count {
        let start = frame_idx * frame_len;
        let end = (start + frame_len).min(samples.len());
        let full_rms = rms(&samples[start..end]);
        let high_rms = rms(&high[start..end]);
        let target_high = (full_rms * HIGHBAND_LEVELER_RATIO + HIGHBAND_LEVELER_FLOOR)
            .min(HIGHBAND_LEVELER_MAX_RMS)
            .max(HIGHBAND_LEVELER_FLOOR);

        let gain = if high_rms > target_high {
            leveled_frames += 1;
            (target_high / high_rms).clamp(HIGHBAND_LEVELER_MIN_GAIN, 1.0)
        } else {
            1.0
        };
        target_gains.push(gain);
    }

    let mut previous_gain = 1.0f32;
    for (frame_idx, &target_gain) in target_gains.iter().enumerate() {
        let start = frame_idx * frame_len;
        let end = (start + frame_len).min(samples.len());
        let span = (end - start).max(1) as f32;
        for i in 0..(end - start) {
            let t = i as f32 / span;
            let gain = if target_gain < previous_gain {
                target_gain + (previous_gain - target_gain) * (1.0 - t).powi(4)
            } else {
                previous_gain + (target_gain - previous_gain) * t
            };
            let idx = start + i;
            samples[idx] = low[idx] + high[idx] * gain;
        }
        previous_gain = target_gain;
    }

    leveled_frames
}

fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| pair[0].is_sign_negative() != pair[1].is_sign_negative())
        .count();
    crossings as f32 / (samples.len() - 1) as f32
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

    #[test]
    fn adaptive_background_gate_reduces_gaps_and_preserves_speech() {
        let sample_rate = 48_000;
        let mut samples = vec![0.006; sample_rate as usize / 10];
        samples.extend(
            sine(sample_rate as usize / 10, sample_rate, 900.0)
                .into_iter()
                .map(|s| s * 0.05),
        );
        samples.extend(vec![0.006; sample_rate as usize / 10]);

        let quiet_before = rms(&samples[..sample_rate as usize / 10]);
        let speech_start = sample_rate as usize / 10;
        let speech_end = speech_start + sample_rate as usize / 10;
        let speech_before = rms(&samples[speech_start..speech_end]);

        let threshold = apply_adaptive_background_gate(
            &mut samples,
            sample_rate,
            BACKGROUND_GATE_FLOOR_GAIN,
            1.4,
            0.0035,
            0.020,
        );

        let quiet_after = rms(&samples[..sample_rate as usize / 10]);
        let speech_after = rms(&samples[speech_start..speech_end]);
        assert!(threshold > 0.0);
        assert!(quiet_after < quiet_before * 0.55);
        assert!(speech_after > speech_before * 0.85);
    }

    #[test]
    fn clone_reference_polish_reduces_background() {
        let sample_rate = 16_000;
        let mut samples = vec![0.012; sample_rate as usize / 5];
        samples.extend(
            sine(sample_rate as usize / 5, sample_rate, 1_000.0)
                .into_iter()
                .map(|s| s * 0.08),
        );

        let report = polish_clone_reference_audio(&mut samples, sample_rate);
        assert!(report.background_gate_threshold > 0.0);
        assert!(report.quiet_rms_after < report.quiet_rms_before * 0.75);
        assert!(peak(&samples) <= PCM_HEADROOM + 1e-6);
    }

    #[test]
    fn clone_reference_fixture_matrix_covers_clean_noisy_and_long() {
        let sample_rate = 16_000;

        let mut clean = clone_reference_fixture(sample_rate, 0.0015, false);
        let clean_speech_before =
            rms(&clean[sample_rate as usize / 5..sample_rate as usize * 2 / 5]);
        let clean_report = polish_clone_reference_audio(&mut clean, sample_rate);
        let clean_speech_after =
            rms(&clean[sample_rate as usize / 5..sample_rate as usize * 2 / 5]);
        assert!(clean_report.background_gate_threshold > 0.0);
        assert!(clean_speech_after > clean_speech_before * 0.80);
        assert!(peak(&clean) <= PCM_HEADROOM + 1e-6);

        let mut noisy = clone_reference_fixture(sample_rate, 0.014, true);
        let noisy_quiet_before = quiet_rms(&noisy, sample_rate);
        let noisy_speech_before =
            rms(&noisy[sample_rate as usize / 5..sample_rate as usize * 2 / 5]);
        let noisy_report = polish_clone_reference_audio(&mut noisy, sample_rate);
        let noisy_speech_after =
            rms(&noisy[sample_rate as usize / 5..sample_rate as usize * 2 / 5]);
        assert!(noisy_report.quiet_rms_before >= noisy_quiet_before * 0.95);
        assert!(noisy_report.quiet_rms_after < noisy_report.quiet_rms_before * 0.80);
        assert!(noisy_speech_after > noisy_speech_before * 0.65);

        let mut long = vec![0.0; sample_rate as usize * 35];
        let trim = trim_clone_reference_audio(&mut long, sample_rate).unwrap();
        assert_eq!(trim.original_samples, sample_rate as usize * 35);
        assert_eq!(trim.trimmed_samples, sample_rate as usize * 30);
        assert_eq!(long.len(), trim.trimmed_samples);
        assert_eq!(trim.sample_rate, sample_rate);
        assert!((trim.original_duration_sec - 35.0).abs() < 1e-6);
        assert_eq!(trim.max_duration_sec, MAX_CLONE_REFERENCE_SECS);
    }

    #[test]
    fn harsh_midband_smoother_softens_noisy_fricative_frames() {
        let sample_rate = 48_000;
        let len = sample_rate as usize / 10;
        let mut samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let voice = (2.0 * std::f32::consts::PI * 900.0 * t).sin() * 0.025;
                let harsh = if (i / 2) % 2 == 0 { 0.018 } else { -0.018 };
                voice + harsh
            })
            .collect();
        let mut softened_ref = samples.clone();
        apply_one_pole_lowpass_zero_phase(
            &mut softened_ref,
            sample_rate,
            HARSH_SMOOTHER_LOWPASS_HZ,
        );
        let diff_before = rms(&samples
            .iter()
            .zip(&softened_ref)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>());

        let frames = apply_harsh_midband_smoother(&mut samples, sample_rate);
        let diff_after = rms(&samples
            .iter()
            .zip(&softened_ref)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>());

        assert!(frames > 0);
        assert!(diff_after < diff_before * 0.75);
    }

    #[test]
    fn patch_boundary_smoother_softens_fm_like_switches() {
        let sample_rate = 48_000;
        let patch_len =
            ((sample_rate as f32 * PATCH_BOUNDARY_SMOOTHER_PATCH_MS) / 1000.0).round() as usize;
        let len = patch_len * 3;
        let mut samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let voice = (2.0 * std::f32::consts::PI * 800.0 * t).sin() * 0.035;
                let tuning_noise = if i >= patch_len && i < patch_len * 2 {
                    if (i / 2) % 2 == 0 {
                        0.018
                    } else {
                        -0.018
                    }
                } else {
                    0.0
                };
                voice + tuning_noise
            })
            .collect();
        let speech_before = rms(&samples);
        let mut high_ref = samples.clone();
        apply_one_pole_highpass_zero_phase(
            &mut high_ref,
            sample_rate,
            PATCH_BOUNDARY_SMOOTHER_LOWPASS_HZ,
        );
        let high_before = rms(&high_ref[patch_len..patch_len + sample_rate as usize / 100]);

        let boundaries = apply_patch_boundary_smoother(&mut samples, sample_rate);

        let mut high_after_ref = samples.clone();
        apply_one_pole_highpass_zero_phase(
            &mut high_after_ref,
            sample_rate,
            PATCH_BOUNDARY_SMOOTHER_LOWPASS_HZ,
        );
        let high_after = rms(&high_after_ref[patch_len..patch_len + sample_rate as usize / 100]);
        let speech_after = rms(&samples);

        assert!(boundaries > 0);
        assert!(
            high_after < high_before * 0.80,
            "high_after={high_after:.6} high_before={high_before:.6} ratio={:.3}",
            high_after / high_before
        );
        assert!(
            speech_after > speech_before * 0.90,
            "speech_after={speech_after:.6} speech_before={speech_before:.6}"
        );
    }

    #[test]
    fn highband_leveler_reduces_switching_background_without_muting_voice() {
        let sample_rate = 48_000;
        let frame_len = (sample_rate as usize * 20) / 1000;
        let len = frame_len * 8;
        let mut samples: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let voice = (2.0 * std::f32::consts::PI * 850.0 * t).sin() * 0.045;
                let frame = i / frame_len;
                let switching_hiss = if frame == 2 || frame == 5 {
                    if (i / 2) % 2 == 0 {
                        0.024
                    } else {
                        -0.024
                    }
                } else {
                    if (i / 2) % 2 == 0 {
                        0.004
                    } else {
                        -0.004
                    }
                };
                voice + switching_hiss
            })
            .collect();

        let voice_before = rms(&samples);
        let mut high_before_ref = samples.clone();
        apply_one_pole_highpass_zero_phase(
            &mut high_before_ref,
            sample_rate,
            HIGHBAND_LEVELER_SPLIT_HZ,
        );
        let burst_before = rms(&high_before_ref[frame_len * 2..frame_len * 3]);

        let frames = apply_highband_residual_leveler(&mut samples, sample_rate);

        let mut high_after_ref = samples.clone();
        apply_one_pole_highpass_zero_phase(
            &mut high_after_ref,
            sample_rate,
            HIGHBAND_LEVELER_SPLIT_HZ,
        );
        let burst_after = rms(&high_after_ref[frame_len * 2..frame_len * 3]);
        let voice_after = rms(&samples);

        assert!(frames >= 2);
        assert!(
            burst_after < burst_before * 0.65,
            "burst_after={burst_after:.6} burst_before={burst_before:.6}"
        );
        assert!(
            voice_after > voice_before * 0.82,
            "voice_after={voice_after:.6} voice_before={voice_before:.6}"
        );
    }

    fn sine(len: usize, sample_rate: u32, hz: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * hz * t).sin()
            })
            .collect()
    }

    fn clone_reference_fixture(sample_rate: u32, noise_floor: f32, add_hiss: bool) -> Vec<f32> {
        let quiet_len = sample_rate as usize / 5;
        let speech_len = sample_rate as usize / 5;
        let mut samples = vec![noise_floor; quiet_len];
        samples.extend((0..speech_len).map(|i| {
            let t = i as f32 / sample_rate as f32;
            let speech = (2.0 * std::f32::consts::PI * 1_000.0 * t).sin() * 0.08;
            let hiss = if add_hiss && i % 2 == 0 {
                noise_floor * 0.6
            } else {
                0.0
            };
            speech + noise_floor + hiss
        }));
        samples.extend(vec![noise_floor; quiet_len]);
        samples
    }
}
