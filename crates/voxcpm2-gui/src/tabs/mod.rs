//! Tab modules for VoxCPM2 egui GUI.

pub mod clone;
pub mod diagnostics;
pub mod model;
pub mod output;
pub mod synth;

use eframe::egui;
use std::path::PathBuf;

pub(crate) fn metrics_path_for_output(output_path: &str) -> PathBuf {
    let mut path = PathBuf::from(output_path);
    path.set_extension("metrics.json");
    path
}

/// All available tabs in the GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Model,
    Synthesis,
    Cloning,
    Output,
    Diagnostics,
}

impl Tab {
    pub fn name(&self) -> &'static str {
        match self {
            Tab::Model => "📦 Model",
            Tab::Synthesis => "🎤 Synthesis",
            Tab::Cloning => "🎭 Cloning",
            Tab::Output => "🔊 Output",
            Tab::Diagnostics => "🔧 Diagnostics",
        }
    }

    /// Show tab bar and return the newly selected tab (or None if unchanged).
    pub fn ui_bar(ui: &mut egui::Ui, active: &mut Tab) {
        ui.horizontal(|ui| {
            for tab in &ALL_TABS {
                let is_active = *tab == *active;
                let btn = egui::Button::new(tab.name()).fill(if is_active {
                    ui.style().visuals.widgets.active.bg_fill
                } else {
                    ui.style().visuals.window_fill()
                });
                if ui.add(btn).clicked() {
                    *active = *tab;
                }
            }
        });
    }
}

const ALL_TABS: [Tab; 5] = [
    Tab::Model,
    Tab::Synthesis,
    Tab::Cloning,
    Tab::Output,
    Tab::Diagnostics,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_path_tracks_wav_output() {
        assert_eq!(
            metrics_path_for_output("output/gui_clone.wav"),
            PathBuf::from("output/gui_clone.metrics.json")
        );
    }
}
