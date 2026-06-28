//! Cloning tab — voice cloning UI scaffold.
//!
//! Voice cloning requires AudioVAE encoder + reference audio pipeline,
//! which is not yet implemented in the Rust backend. This tab provides
//! the UI skeleton so users can see the planned interface.

use eframe::egui;

#[derive(Default)]
pub struct CloneTab {
    /// Path to reference WAV file.
    pub ref_audio_path: String,
    /// Optional transcript of the reference audio.
    pub ref_transcript: String,
    /// Similarity / style control strength (0.0 – 1.0).
    pub similarity: f32,
    /// Output file path.
    pub output_path: String,
    /// Status message shown to the user.
    pub status: String,
}

impl CloneTab {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Voice Cloning");

        // ── Reference audio ──
        ui.horizontal(|ui| {
            ui.label("Reference audio:");
            ui.text_edit_singleline(&mut self.ref_audio_path);
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Audio", &["wav", "mp3", "flac", "ogg"])
                    .set_file_name("reference.wav")
                    .pick_file()
                {
                    self.ref_audio_path = path.display().to_string();
                }
            }
        });

        // ── Reference transcript ──
        ui.label("Reference transcript (optional):");
        ui.text_edit_multiline(&mut self.ref_transcript);

        // ── Parameters ──
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut self.similarity, 0.0..=1.0)
                    .text("Similarity"),
            );
        });

        // ── Output path ──
        ui.horizontal(|ui| {
            ui.label("Output file:");
            ui.text_edit_singleline(&mut self.output_path);
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV", &["wav"])
                    .set_file_name("clone.wav")
                    .save_file()
                {
                    self.output_path = path.display().to_string();
                }
            }
        });

        ui.separator();

        // ── Clone button (disabled — backend not ready) ──
        ui.horizontal(|ui| {
            ui.add_enabled(
                false,
                egui::Button::new("Clone Voice")
                    .fill(egui::Color32::from_rgb(80, 80, 80)),
            );
            ui.label("(not yet implemented)");
        });

        // ── Explanation ──
        ui.separator();
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(30, 30, 40))
            .show(ui, |ui| {
                ui.label("Voice cloning requires several backend components");
                ui.label("not yet available in this Rust build:");
                ui.label("  • AudioVAE encoder (encode audio → latent)");
                ui.label("  • Reference audio pipeline (load, resample, align)");
                ui.label("  • Special token construction (ref_audio tokens)");
                ui.label("  • feat_encoder + fusion_concat_proj integration");
                ui.label("");
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "Use the Python voxcpm CLI for voice cloning:",
                );
                ui.label("  voxcpm clone --reference-audio REF.wav \\");
                ui.label("    --text \"目標文字\" --output out.wav");
            });

        // ── Status ──
        if !self.status.is_empty() {
            ui.separator();
            ui.label(&self.status);
        }
    }
}
