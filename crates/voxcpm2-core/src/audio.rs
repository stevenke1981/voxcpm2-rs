use std::path::Path;

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_tone_is_valid() {
        let s = smoke_tone("hello", 48_000);
        check_audio(&s).unwrap();
        assert!(s.len() > 10_000);
    }
}
