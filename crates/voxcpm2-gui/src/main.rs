use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use std::{path::PathBuf, thread};
use voxcpm2_core::{SynthRequest, VoxPipeline};

#[derive(Debug)]
enum GuiCommand {
    Generate(SynthRequest),
}

#[derive(Debug)]
enum GuiEvent {
    Log(String),
    Finished(PathBuf),
    Failed(String),
}

struct VoxApp {
    model_dir: String,
    text: String,
    output: String,
    device: String,
    cfg: f32,
    steps: usize,
    dry_run: bool,
    log: Vec<String>,
    tx: Sender<GuiCommand>,
    rx: Receiver<GuiEvent>,
}

impl Default for VoxApp {
    fn default() -> Self {
        let (cmd_tx, cmd_rx) = unbounded::<GuiCommand>();
        let (evt_tx, evt_rx) = unbounded::<GuiEvent>();
        thread::spawn(move || worker_loop(cmd_rx, evt_tx));
        Self {
            model_dir: "models/VoxCPM2".into(),
            text: "你好，這是 VoxCPM2 Rust Candle egui 測試。".into(),
            output: "output/gui_smoke.wav".into(),
            device: "auto".into(),
            cfg: 2.0,
            steps: 10,
            dry_run: true,
            log: vec!["Ready".into()],
            tx: cmd_tx,
            rx: evt_rx,
        }
    }
}

fn worker_loop(rx: Receiver<GuiCommand>, tx: Sender<GuiEvent>) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            GuiCommand::Generate(req) => {
                let _ = tx.send(GuiEvent::Log("Generating...".into()));
                let result = (|| -> anyhow::Result<PathBuf> {
                    let mut pipe =
                        VoxPipeline::new(&req.device, req.model_dir.as_deref(), req.dry_run)?;
                    let out = pipe.synthesize(&req)?.output_path;
                    Ok(out)
                })();
                match result {
                    Ok(path) => {
                        let _ = tx.send(GuiEvent::Finished(path));
                    }
                    Err(e) => {
                        let _ = tx.send(GuiEvent::Failed(e.to_string()));
                    }
                }
            }
        }
    }
}

impl eframe::App for VoxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(evt) = self.rx.try_recv() {
            match evt {
                GuiEvent::Log(s) => self.log.push(s),
                GuiEvent::Finished(path) => self.log.push(format!("Finished: {}", path.display())),
                GuiEvent::Failed(e) => self.log.push(format!("Error: {e}")),
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("VoxCPM2 Rust Candle");
                ui.label(format!("device: {}", self.device));
                ui.checkbox(&mut self.dry_run, "dry-run smoke");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Synthesis");
            ui.horizontal(|ui| {
                ui.label("Model dir");
                ui.text_edit_singleline(&mut self.model_dir);
                if ui.button("Browse").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.model_dir = path.display().to_string();
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Output");
                ui.text_edit_singleline(&mut self.output);
            });
            ui.horizontal(|ui| {
                ui.label("Device");
                ui.text_edit_singleline(&mut self.device);
                ui.add(egui::Slider::new(&mut self.cfg, 0.5..=5.0).text("cfg"));
                ui.add(egui::Slider::new(&mut self.steps, 1..=50).text("steps"));
            });
            ui.label("Text / Voice Design");
            ui.text_edit_multiline(&mut self.text);
            if ui.button("Generate").clicked() {
                let req = SynthRequest {
                    text: self.text.clone(),
                    model_dir: Some(PathBuf::from(self.model_dir.clone())),
                    output_path: PathBuf::from(self.output.clone()),
                    device: self.device.clone(),
                    cfg_value: self.cfg,
                    inference_timesteps: self.steps,
                    dry_run: self.dry_run,
                    label_ai_generated: true,
                };
                let _ = self.tx.send(GuiCommand::Generate(req));
            }

            ui.separator();
            ui.heading("Diagnostics");
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .show(ui, |ui| {
                    for line in &self.log {
                        ui.label(line);
                    }
                });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "VoxCPM2 Rust Candle egui",
        native_options,
        Box::new(|_cc| Ok(Box::<VoxApp>::default())),
    )
}
