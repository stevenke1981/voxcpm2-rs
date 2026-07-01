//! Diagnostics tab — device info, VRAM estimate, log output.

use eframe::egui;

#[derive(Default)]
pub struct DiagnosticsTab {
    pub log: Vec<String>,
}

impl DiagnosticsTab {
    pub fn ui(&mut self, ui: &mut egui::Ui, model_dir: &str, device_str: &str) {
        ui.heading("Diagnostics");

        // ── Device info ──
        ui.horizontal(|ui| {
            ui.label("Configured device:");
            ui.colored_label(
                egui::Color32::YELLOW,
                if device_str.is_empty() {
                    "auto"
                } else {
                    device_str
                },
            );
        });
        if let Ok(info) = probe_device(device_str) {
            ui.colored_label(egui::Color32::GREEN, &info);
        }

        ui.separator();

        // ── VRAM estimate ──
        ui.label("VRAM estimate (model only):");
        ui.label("  2.3B params @ BF16 ≈ 4.6 GB");
        ui.label("  2.3B params @ F32 ≈ 9.2 GB");
        ui.label("  AudioVAE @ F32 ≈ 0.4 GB");
        ui.label("  KV cache (4K tokens, 28 layers) ≈ 0.6 GB");
        ui.label("  Total estimate: ~5.6 GB (BF16) / ~10.2 GB (F32)");

        if device_str.contains("cuda") {
            ui.colored_label(egui::Color32::YELLOW, "Minimum: 8GB VRAM GPU recommended");
        }

        ui.separator();

        // ── Model info ──
        ui.label("Model directory:");
        ui.label(if model_dir.is_empty() {
            "(not set)".to_string()
        } else {
            model_dir.to_string()
        });

        ui.separator();

        // ── Log ──
        ui.heading("Event log");
        egui::ScrollArea::vertical()
            .max_height(300.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in &self.log {
                    ui.label(line);
                }
                if self.log.is_empty() {
                    ui.label("(no events)");
                }
            });

        ui.separator();
        if ui.button("Clear log").clicked() {
            self.log.clear();
        }
    }
}

fn probe_device(device_str: &str) -> Result<String, String> {
    let pref = match voxcpm2_core::device::DevicePreference::parse(device_str) {
        Ok(p) => p,
        Err(e) => return Err(format!("Parse error: {e}")),
    };
    match voxcpm2_core::device::select_device(pref) {
        Ok(dev) => Ok(format!("Active: {dev:?}")),
        Err(e) => Err(format!("Error: {e}")),
    }
}
