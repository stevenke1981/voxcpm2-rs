//! Tab modules for VoxCPM2 egui GUI.

pub mod clone;
pub mod diagnostics;
pub mod model;
pub mod output;
pub mod synth;

use eframe::egui;

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
                let btn = egui::Button::new(tab.name())
                    .fill(if is_active {
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
