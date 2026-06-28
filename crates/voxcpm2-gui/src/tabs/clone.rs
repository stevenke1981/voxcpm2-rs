//! Cloning tab — voice cloning UI with full backend integration.
//!
//! Uses AudioVAE encoder + reference audio pipeline to clone voices.

use eframe::egui;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use voxcpm2_core::SynthRequest;

pub struct CloneTab {
    /// Path to reference WAV file.
    pub ref_audio_path: String,
    /// Text to synthesize (what the cloned voice will say).
    pub text: String,
    /// Optional transcript of the reference audio (for future alignment).
    pub ref_transcript: String,
    /// Similarity / style control strength (0.0 – 1.0). Default 1.0 = full clone.
    pub similarity: f32,
    /// Output file path.
    pub output_path: String,
    /// CFG scale.
    pub cfg: f32,
    /// Inference steps.
    pub steps: usize,
    /// Random seed as string (empty = random).
    pub seed: String,
    /// Timestep scheduler.
    pub t_scheduler: String,
    /// Status message shown to the user.
    pub status: String,
    /// Disables the clone button while generating.
    pub generate_disabled: bool,
    /// Set to Some(true) when user clicks Clone Voice.
    pub pending_generate: Option<SynthRequest>,
}

impl Default for CloneTab {
    fn default() -> Self {
        Self {
            similarity: 1.0,
            steps: 30,
            cfg: 2.5,
            t_scheduler: "uniform".into(),
            ref_audio_path: String::new(),
            text: String::new(),
            ref_transcript: String::new(),
            output_path: String::new(),
            seed: String::new(),
            status: String::new(),
            generate_disabled: false,
            pending_generate: None,
        }
    }
}

impl CloneTab {
    pub fn ui(&mut self, ui: &mut egui::Ui, model_dir: &str, device_str: &str, cancel_flag: &Arc<AtomicBool>) {
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

        // ── Text to synthesize ──
        ui.label("Text to synthesize (what the cloned voice will say):");
        ui.text_edit_multiline(&mut self.text);

        // ── Reference transcript (optional) ──
        ui.label("Reference transcript (optional, for future alignment):");
        ui.text_edit_multiline(&mut self.ref_transcript);

        // ── Parameters ──
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.cfg, 0.5..=10.0).text("CFG scale"));
            ui.add(egui::Slider::new(&mut self.steps, 1..=100).text("Steps"));
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut self.similarity, 0.0..=1.0)
                    .text("Clone strength"),
            );
            ui.label("(0=text-only, 1=full clone)");
        });
        ui.horizontal(|ui| {
            ui.label("Seed (optional):");
            ui.text_edit_singleline(&mut self.seed);
        });
        ui.horizontal(|ui| {
            ui.label("Scheduler:");
            ui.selectable_value(&mut self.t_scheduler, "uniform".into(), "Uniform");
            ui.selectable_value(&mut self.t_scheduler, "log-norm".into(), "Log-Norm");
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

        // ── Clone / Cancel button ──
        ui.horizontal(|ui| {
            if self.generate_disabled {
                if ui
                    .add(
                        egui::Button::new("⏹ Cancel")
                            .fill(egui::Color32::from_rgb(180, 40, 40))
                            .min_size(egui::vec2(120.0, 30.0)),
                    )
                    .clicked()
                {
                    cancel_flag.store(true, Ordering::SeqCst);
                }
            } else {
                let has_ref = !self.ref_audio_path.is_empty() && PathBuf::from(&self.ref_audio_path).exists();
                let has_text = !self.text.trim().is_empty();
                let enabled = has_ref && has_text;
                let tip = if !has_ref {
                    "Select a reference audio WAV file"
                } else if !has_text {
                    "Enter text for the cloned voice to speak"
                } else {
                    "Generate cloned speech"
                };
                let btn = egui::Button::new("Clone Voice")
                    .fill(if enabled {
                        egui::Color32::from_rgb(0, 120, 200)
                    } else {
                        egui::Color32::from_rgb(80, 80, 80)
                    });
                if ui.add_enabled(enabled, btn).on_hover_text(tip).clicked() {
                    let ref_path = std::path::PathBuf::from(&self.ref_audio_path);
                    let transcript = if self.ref_transcript.is_empty() {
                        None
                    } else {
                        Some(self.ref_transcript.clone())
                    };
                    let seed_parsed = self.seed.trim().parse::<u64>().ok();
                    let req = SynthRequest {
                        text: self.text.clone(),
                        voice_design: None,
                        model_dir: Some(std::path::PathBuf::from(model_dir)),
                        output_path: std::path::PathBuf::from(&self.output_path),
                        device: device_str.to_string(),
                        cfg_value: self.cfg,
                        inference_timesteps: self.steps,
                        seed: seed_parsed,
                        post_gain: None,
                        dry_run: false,
                        label_ai_generated: true,
                        max_autoregressive_steps: None,
                        t_scheduler: self.t_scheduler.clone(),
                        latent_norm_scale: None,
                        ref_audio_path: Some(ref_path),
                        ref_transcript: transcript,
                        clone_strength: self.similarity as f64,
                    };
                    self.pending_generate = Some(req);
                    self.generate_disabled = true;
                }
            }
        });

        ui.separator();

        // ── Explanation ──
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(30, 30, 40))
            .show(ui, |ui| {
                ui.label("Voice clone: reference audio → AudioVAE encoder → latent");
                ui.label("patches injected into the autoregressive loop as a speaker");
                ui.label("conditioning prefix.");
            });

        // ── Status ──
        if !self.status.is_empty() {
            ui.separator();
            ui.label(&self.status);
        }
    }
}
