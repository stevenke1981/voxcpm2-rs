//! Model tab — model directory selection, inspect, manifest display.

use eframe::egui;
use std::path::PathBuf;
use voxcpm2_core::{full_asset_report, AssetReport, SafetensorsManifest};

#[derive(Default)]
pub struct ModelTab {
    pub model_dir: String,
    pub device_str: String,
    pub manifest_info: Option<String>,
    pub safetensors_info: Option<String>,
    pub error: Option<String>,
}

impl ModelTab {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Model & Device");

        // ── Model directory ──
        ui.horizontal(|ui| {
            ui.label("Model directory:");
            ui.text_edit_singleline(&mut self.model_dir);
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.model_dir = path.display().to_string();
                }
            }
        });

        // ── Device ──
        ui.horizontal(|ui| {
            ui.label("Device:");
            ui.text_edit_singleline(&mut self.device_str);
            if ui.button("Auto-detect").clicked() {
                self.device_str = "auto".to_string();
            }
        });

        // ── Inspect button ──
        if ui.button("Inspect Model").clicked() {
            self.run_inspect();
        }
        ui.separator();

        // ── Results ──
        if let Some(err) = &self.error {
            ui.colored_label(egui::Color32::RED, err);
        }
        if let Some(mi) = &self.manifest_info {
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    ui.label(mi);
                });
        }
        if let Some(st) = &self.safetensors_info {
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    ui.label(st);
                });
        }

        // ── Quick model checks ──
        ui.separator();
        ui.label("Required files:");
        let model_dir = PathBuf::from(&self.model_dir);
        for (name, status) in check_files(&model_dir) {
            let icon = match status {
                FileStatus::Present => "✅",
                FileStatus::Missing => "❌",
            };
            ui.label(format!("  {icon} {name}"));
        }
    }

    fn run_inspect(&mut self) {
        let model_dir = PathBuf::from(&self.model_dir);
        self.error = None;
        self.manifest_info = None;
        self.safetensors_info = None;

        if !model_dir.exists() {
            self.error = Some(format!("Directory not found: {}", model_dir.display()));
            return;
        }

        let st_path = model_dir.join("model.safetensors");
        if !st_path.exists() {
            self.error = Some("model.safetensors not found".into());
            // Show assets report anyway
            self.show_asset_report(&model_dir);
            return;
        }

        // Inspect safetensors
        let manifest = voxcpm2_core::inspect_safetensors(&st_path);
        self.safetensors_info = Some(format!(
            "model.safetensors:\n  Tensors: {}\n  Total params: {}\n  Dtypes: {:?}",
            manifest.num_tensors.unwrap_or(0),
            manifest.total_params.unwrap_or(0),
            manifest.dtype_counts.as_ref().map(|d| d.iter().map(|(k,v)| format!("{k}:{v}")).collect::<Vec<_>>().join(", ")).unwrap_or_default(),
        ));

        self.show_asset_report(&model_dir);
    }

    fn show_asset_report(&mut self, model_dir: &PathBuf) {
        let report = full_asset_report(model_dir, false).unwrap_or_else(|_| {
            AssetReport {
                files: vec![],
                manifest: SafetensorsManifest {
                    path: model_dir.join("model.safetensors"),
                    exists: false,
                    num_tensors: None,
                    total_params: None,
                    dtype_counts: None,
                    tensor_names: None,
                },
                audiovae_manifest: None,
                fixes: vec![],
                model_ready: false,
            }
        });
        let mut lines = Vec::new();
        for f in &report.files {
            lines.push(format!("  {}  {}", if f.exists { "✅" } else { "❌" }, f.path.display()));
        }
        self.manifest_info = Some(lines.join("\n"));
    }
}

#[derive(Debug)]
enum FileStatus {
    Present,
    Missing,
}

fn check_files(dir: &std::path::Path) -> Vec<(&'static str, FileStatus)> {
    let checks: &[&str] = &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "audiovae.safetensors",
    ];
    checks
        .iter()
        .map(|name| {
            let status = if dir.join(name).exists() {
                FileStatus::Present
            } else {
                FileStatus::Missing
            };
            (*name, status)
        })
        .collect()
}
