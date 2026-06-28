//! Output tab — playback controls, waveform preview, export.

use eframe::egui;
use rodio::{buffer::SamplesBuffer, OutputStream, Sink};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

#[derive(Default)]
pub struct OutputTab {
    pub audio_samples: Vec<f32>,
    pub audio_sample_rate: u32,
    pub is_playing: Arc<AtomicBool>,
    pub output_path: String,
    pub status: String,
}

impl OutputTab {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Output");

        let has_audio = !self.audio_samples.is_empty();

        // ── Output path ──
        ui.horizontal(|ui| {
            ui.label("Export path:");
            ui.text_edit_singleline(&mut self.output_path);
        });

        if has_audio {
            ui.separator();

            // ── Info ──
            let duration = self.audio_samples.len() as f64 / self.audio_sample_rate as f64;
            let peak = self
                .audio_samples
                .iter()
                .map(|s| s.abs())
                .fold(0.0f32, f32::max);
            ui.label(format!(
                "{:.1}s @ {}Hz — peak: {:.4}",
                duration, self.audio_sample_rate, peak
            ));

            // ── Playback ──
            ui.horizontal(|ui| {
                let playing = self.is_playing.load(Ordering::SeqCst);
                let btn_label = if playing { "⏹ Stop" } else { "▶ Play" };
                if ui
                    .add_sized([80.0, 30.0], egui::Button::new(btn_label))
                    .clicked()
                {
                    if playing {
                        self.is_playing.store(false, Ordering::SeqCst);
                    } else {
                        self.play_audio();
                    }
                }

                if ui
                    .add_sized([80.0, 30.0], egui::Button::new("💾 Export WAV"))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("WAV", &["wav"])
                        .set_file_name("output.wav")
                        .save_file()
                    {
                        let path_str = path.display().to_string();
                        self.output_path = path_str.clone();
                        self.export_wav(&path_str);
                    }
                }
            });

            ui.separator();

            // ── Waveform ──
            self.waveform_widget(ui);

            // ── Status ──
            if !self.status.is_empty() {
                ui.label(&self.status);
            }
        } else {
            ui.label("No audio generated yet. Use the Synthesis tab to generate speech.");
        }
    }

    fn waveform_widget(&self, ui: &mut egui::Ui) {
        let total = self.audio_samples.len();
        if total == 0 {
            return;
        }

        let max_f = self
            .audio_samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
            .max(1e-8);

        let painter = ui.painter();
        let rect = ui.available_rect_before_wrap();
        let w = (rect.width() - 10.0).min(800.0);
        let h = 80.0;
        let origin = egui::pos2(rect.left() + 5.0, rect.top());

        // Draw center line
        painter.line_segment(
            [
                egui::pos2(origin.x, origin.y + h * 0.5),
                egui::pos2(origin.x + w, origin.y + h * 0.5),
            ],
            egui::Stroke::new(0.5, egui::Color32::DARK_GRAY),
        );

        // Draw waveform
        let mut points: Vec<egui::Pos2> = Vec::with_capacity(512);
        let step = (total / 512).max(1);
        for i in (0..total).step_by(step) {
            let x = origin.x + (i as f32 / total as f32) * w;
            let y = origin.y + h * 0.5 - (self.audio_samples[i] / max_f) * h * 0.4;
            points.push(egui::pos2(x, y));
        }
        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 180, 255));
        painter.add(egui::Shape::line(points, stroke));

        // Reserve space
        ui.allocate_space(egui::vec2(w, h));
    }

    fn play_audio(&self) {
        let samples = self.audio_samples.clone();
        let sr = self.audio_sample_rate;
        let flag = self.is_playing.clone();
        flag.store(true, Ordering::SeqCst);
        thread::spawn(move || {
            let (_stream, stream_handle) = match OutputStream::try_default() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Audio output unavailable: {e}");
                    flag.store(false, Ordering::SeqCst);
                    return;
                }
            };
            let sink = match Sink::try_new(&stream_handle) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Audio sink error: {e}");
                    flag.store(false, Ordering::SeqCst);
                    return;
                }
            };

            let clamped: Vec<f32> = samples.iter().map(|&s| s.clamp(-1.0, 1.0)).collect();
            let source = SamplesBuffer::new(1, sr, clamped);
            sink.append(source);
            while !sink.empty() && flag.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(50));
            }
            sink.stop();
            flag.store(false, Ordering::SeqCst);
        });
    }

    fn export_wav(&mut self, path: &str) {
        let result = (|| -> anyhow::Result<()> {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: self.audio_sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(path, spec)?;
            for &s in &self.audio_samples {
                let clamped = s.clamp(-1.0, 1.0);
                let sample = (clamped * i16::MAX as f32) as i16;
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
            Ok(())
        })();
        match result {
            Ok(_) => self.status = format!("Exported to {path}"),
            Err(e) => self.status = format!("Export error: {e}"),
        }
    }
}
