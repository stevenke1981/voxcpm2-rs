//! Synthesis tab — text input, Voice Design, cfg, steps, seed, generate, cancel.

use eframe::egui;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use voxcpm2_core::SynthRequest;

#[derive(Default)]
pub struct SynthTab {
    pub text: String,
    pub output_path: String,
    pub cfg: f32,
    pub steps: usize,
    pub seed: String,
    pub voice_design: String,
    pub dry_run: bool,
    pub generate_disabled: bool,
    pub t_scheduler: String,
    pub latent_norm_scale: Option<f64>,
    /// Set to Some(true) when user clicks generate.
    pub pending_generate: Option<SynthRequest>,
}

impl SynthTab {
    pub fn ui(&mut self, ui: &mut egui::Ui, model_dir: &str, device_str: &str, cancel_flag: &Arc<AtomicBool>) {
        ui.heading("Synthesis");

        ui.horizontal(|ui| {
            ui.label("Output file:");
            ui.text_edit_singleline(&mut self.output_path);
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV", &["wav"])
                    .set_file_name("output.wav")
                    .save_file()
                {
                    self.output_path = path.display().to_string();
                }
            }
        });

        // ── Parameters ──
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.cfg, 0.5..=10.0).text("CFG scale"));
            ui.add(egui::Slider::new(&mut self.steps, 1..=100).text("Steps"));
        });
        ui.horizontal(|ui| {
            ui.label("Seed (optional):");
            ui.text_edit_singleline(&mut self.seed);
            ui.checkbox(&mut self.dry_run, "Dry-run (fake audio)");
        });

        // ── Text input ──
        ui.label("Text to synthesize:");
        ui.text_edit_multiline(&mut self.text);

        // ── Voice Design ──
        ui.label("Voice Design (optional description):");
        ui.text_edit_multiline(&mut self.voice_design);

        // ── Generate / Cancel ──
        ui.separator();
        if self.generate_disabled {
            // Show Cancel button while generation is running
            if ui
                .add(
                    egui::Button::new("⏹ Cancel")
                        .fill(egui::Color32::from_rgb(180, 40, 40))
                        .min_size(egui::vec2(100.0, 30.0)),
                )
                .clicked()
            {
                cancel_flag.store(true, Ordering::SeqCst);
            }
        } else {
            let btn = egui::Button::new("Generate")
                .fill(egui::Color32::from_rgb(0, 120, 200));
            if ui.add(btn).clicked() {
                let seed_parsed = self.seed.trim().parse::<u64>().ok();
                let req = SynthRequest {
                    text: self.text.clone(),
                    voice_design: Some(self.voice_design.clone())
                        .filter(|s| !s.is_empty()),
                    model_dir: Some(std::path::PathBuf::from(model_dir)),
                    output_path: std::path::PathBuf::from(&self.output_path),
                    device: device_str.to_string(),
                    cfg_value: self.cfg,
                    inference_timesteps: self.steps,
                    seed: seed_parsed,
                    post_gain: None,
                    dry_run: self.dry_run,
                    label_ai_generated: true,
                    max_autoregressive_steps: None,
                    t_scheduler: self.t_scheduler.clone(),
                    latent_norm_scale: self.latent_norm_scale,
                    ref_audio_path: None,
                    ref_transcript: None,
                    clone_strength: 1.0,
                };
                self.pending_generate = Some(req);
                self.generate_disabled = true;
            }
        }
    }
}
