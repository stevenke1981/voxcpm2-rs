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

use super::metrics_path_for_output;

pub struct CloneTab {
    /// Path to reference WAV file (for timbre cloning, isolated ref mode).
    pub ref_audio_path: String,
    /// Text to synthesize (what the cloned voice will say).
    pub text: String,
    /// Optional transcript of the reference audio (for future alignment).
    pub ref_transcript: String,
    /// Optional continuation prompt audio (for prompt-only or combined ultimate cloning).
    pub prompt_audio_path: String,
    /// Exact transcript of the prompt audio (required when prompt_audio is provided).
    pub prompt_text: String,
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
    /// User confirmation that the reference voice is authorized (required for any clone using ref or prompt audio).
    pub has_voice_consent: bool,
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
            prompt_audio_path: String::new(),
            prompt_text: String::new(),
            output_path: String::new(),
            seed: String::new(),
            status: String::new(),
            has_voice_consent: false,
            generate_disabled: false,
            pending_generate: None,
        }
    }
}

impl CloneTab {
    fn has_reference_audio(&self) -> bool {
        !self.ref_audio_path.is_empty() && PathBuf::from(&self.ref_audio_path).exists()
    }

    fn has_prompt_audio(&self) -> bool {
        !self.prompt_audio_path.is_empty() && PathBuf::from(&self.prompt_audio_path).exists()
    }

    fn has_prompt_transcript(&self) -> bool {
        !self.prompt_text.trim().is_empty()
    }

    fn has_synthesis_text(&self) -> bool {
        !self.text.trim().is_empty()
    }

    fn can_generate(&self) -> bool {
        let has_conditioning = self.has_reference_audio() || self.has_prompt_audio();
        let prompt_ok = if self.has_prompt_audio() {
            self.has_prompt_transcript()
        } else {
            true
        };
        has_conditioning && self.has_synthesis_text() && self.has_voice_consent && prompt_ok
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        model_dir: &str,
        device_str: &str,
        cancel_flag: &Arc<AtomicBool>,
    ) {
        ui.heading("Voice Cloning");

        // ── Reference audio (timbre / isolated reference for cloning) ──
        ui.horizontal(|ui| {
            ui.label("Reference audio (timbre):");
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

        // ── Prompt audio + transcript (for continuation / ultimate cloning) ──
        ui.horizontal(|ui| {
            ui.label("Prompt audio (continuation, optional):");
            ui.text_edit_singleline(&mut self.prompt_audio_path);
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Audio", &["wav", "mp3", "flac", "ogg"])
                    .set_file_name("prompt.wav")
                    .pick_file()
                {
                    self.prompt_audio_path = path.display().to_string();
                }
            }
        });
        ui.label("Prompt transcript (exact text of prompt audio; required if prompt audio is provided):");
        ui.text_edit_multiline(&mut self.prompt_text);

        // ── Text to synthesize ──
        ui.label("Text to synthesize (what the cloned voice will say):");
        ui.text_edit_multiline(&mut self.text);

        // ── Reference transcript (optional) ──
        ui.label("Reference transcript (optional, for future alignment):");
        ui.text_edit_multiline(&mut self.ref_transcript);

        if self.has_prompt_audio() && !self.has_reference_audio() {
            ui.label(egui::RichText::new("Prompt-only continuation mode (no separate reference timbre).").italics());
        } else if self.has_prompt_audio() && self.has_reference_audio() {
            ui.label(egui::RichText::new("Combined mode: reference timbre + prompt continuation for highest fidelity.").italics());
        }

        // ── Parameters ──
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.cfg, 0.5..=10.0).text("CFG scale"));
            ui.add(egui::Slider::new(&mut self.steps, 1..=100).text("Steps"));
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.similarity, 0.0..=1.0).text("Clone strength"));
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

        ui.checkbox(
            &mut self.has_voice_consent,
            "I have rights/consent to use this reference voice",
        );

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
                let has_ref = self.has_reference_audio();
                let has_text = self.has_synthesis_text();
                let enabled = self.can_generate();
                let tip = if !self.has_voice_consent {
                    "Confirm voice rights/consent before cloning"
                } else if !has_ref {
                    "Select a reference audio WAV file"
                } else if !has_text {
                    "Enter text for the cloned voice to speak"
                } else {
                    "Generate cloned speech"
                };
                let btn = egui::Button::new("Clone Voice").fill(if enabled {
                    egui::Color32::from_rgb(0, 120, 200)
                } else {
                    egui::Color32::from_rgb(80, 80, 80)
                });
                if ui.add_enabled(enabled, btn).on_hover_text(tip).clicked() {
                    let ref_path = if self.ref_audio_path.is_empty() {
                        None
                    } else {
                        Some(std::path::PathBuf::from(&self.ref_audio_path))
                    };
                    let ref_transcript = if self.ref_transcript.is_empty() {
                        None
                    } else {
                        Some(self.ref_transcript.clone())
                    };
                    let prompt_path = if self.prompt_audio_path.is_empty() {
                        None
                    } else {
                        Some(std::path::PathBuf::from(&self.prompt_audio_path))
                    };
                    let p_text = if self.prompt_text.trim().is_empty() {
                        None
                    } else {
                        Some(self.prompt_text.clone())
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
                        ref_audio_path: ref_path,
                        ref_transcript,
                        prompt_audio_path: prompt_path,
                        prompt_text: p_text,
                        clone_strength: self.similarity as f64,
                        metrics_output_path: Some(metrics_path_for_output(&self.output_path)),
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
                ui.label("Modes supported:");
                ui.label("• Reference only: timbre cloning from ref audio.");
                ui.label("• Prompt only: audio continuation using exact prompt transcript (ultimate cloning).");
                ui.label("• Combined: reference timbre + prompt continuation for max fidelity.");
                ui.label("Provide prompt audio + its exact transcript for continuation. Reference provides isolated timbre.");
            });

        // ── Status ──
        if !self.status.is_empty() {
            ui.separator();
            ui.label(&self.status);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_requires_reference_text_and_consent() -> anyhow::Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let ref_path = std::env::temp_dir().join(format!("voxcpm2-gui-ref-{unique}.wav"));
        std::fs::write(&ref_path, [])?;

        let mut tab = CloneTab {
            ref_audio_path: ref_path.display().to_string(),
            text: "authorized clone test".into(),
            ..CloneTab::default()
        };

        assert!(!tab.can_generate());
        tab.has_voice_consent = true;
        assert!(tab.can_generate());

        let _ = std::fs::remove_file(ref_path);
        Ok(())
    }

    #[test]
    fn clone_supports_prompt_only_and_combined() -> anyhow::Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let ref_path = std::env::temp_dir().join(format!("voxcpm2-gui-ref-{unique}.wav"));
        let prompt_path = std::env::temp_dir().join(format!("voxcpm2-gui-prompt-{unique}.wav"));
        std::fs::write(&ref_path, [])?;
        std::fs::write(&prompt_path, [])?;

        let mut tab = CloneTab {
            ref_audio_path: ref_path.display().to_string(),
            prompt_audio_path: prompt_path.display().to_string(),
            prompt_text: "這是提示音訊的逐字稿。".into(),
            text: "combined clone test".into(),
            has_voice_consent: true,
            ..CloneTab::default()
        };

        assert!(tab.can_generate()); // combined ok

        // prompt-only (no ref)
        tab.ref_audio_path.clear();
        assert!(tab.can_generate());

        // missing prompt transcript should fail
        tab.prompt_text.clear();
        assert!(!tab.can_generate());

        let _ = std::fs::remove_file(&ref_path);
        let _ = std::fs::remove_file(&prompt_path);
        Ok(())
    }
}
