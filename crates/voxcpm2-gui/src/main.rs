use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::{path::PathBuf, thread};
use voxcpm2_core::{SynthRequest, VoxPipeline};

mod tabs;
use tabs::clone::CloneTab;
use tabs::model::ModelTab;
use tabs::output::OutputTab;
use tabs::synth::SynthTab;
use tabs::{diagnostics::DiagnosticsTab, Tab};

#[derive(Debug)]
enum GuiCommand {
    Generate(SynthRequest),
}

#[derive(Debug)]
enum GuiEvent {
    Finished {
        path: PathBuf,
        samples: Vec<f32>,
        sample_rate: u32,
    },
    Failed(String),
}

struct VoxApp {
    active_tab: Tab,
    model_tab: ModelTab,
    synth_tab: SynthTab,
    clone_tab: CloneTab,
    output_tab: OutputTab,
    diagnostics_tab: DiagnosticsTab,
    tx: Sender<GuiCommand>,
    rx: Receiver<GuiEvent>,
    cancel_flag: Arc<AtomicBool>,
}

impl Default for VoxApp {
    fn default() -> Self {
        let (cmd_tx, cmd_rx) = unbounded::<GuiCommand>();
        let (evt_tx, evt_rx) = unbounded::<GuiEvent>();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        thread::spawn({
            let cancel_flag = cancel_flag.clone();
            move || worker_loop(cmd_rx, evt_tx, cancel_flag)
        });
        Self {
            active_tab: Tab::Synthesis,
            model_tab: ModelTab {
                model_dir: "models/VoxCPM2".into(),
                device_str: "auto".into(),
                ..Default::default()
            },
            synth_tab: SynthTab {
                text: "你好，這是 VoxCPM2 Rust Candle egui 測試。".into(),
                output_path: "output/gui_synth.wav".into(),
                cfg: 2.0,
                steps: 30,
                ..Default::default()
            },
            clone_tab: CloneTab {
                output_path: "output/gui_clone.wav".into(),
                ..Default::default()
            },
            output_tab: OutputTab {
                output_path: "output/gui_synth.wav".into(),
                ..Default::default()
            },
            diagnostics_tab: DiagnosticsTab::default(),
            tx: cmd_tx,
            rx: evt_rx,
            cancel_flag,
        }
    }
}

fn worker_loop(rx: Receiver<GuiCommand>, tx: Sender<GuiEvent>, cancel_flag: Arc<AtomicBool>) {
    while let Ok(cmd) = rx.recv() {
        // Reset cancel flag for each new command
        cancel_flag.store(false, Ordering::SeqCst);

        match cmd {
            GuiCommand::Generate(req) => {
                let result = (|| -> anyhow::Result<(PathBuf, Vec<f32>, u32)> {
                    let dry_run = req.dry_run;
                    let mut pipe =
                        VoxPipeline::new(&req.device, req.model_dir.as_deref(), dry_run)?;
                    let out = pipe.synthesize(&req, Some(&cancel_flag))?;
                    let sr = out.sample_rate;

                    // Read WAV for playback
                    let mut reader = hound::WavReader::open(&out.output_path)?;
                    let samples: Vec<f32> = reader
                        .samples::<i16>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / i16::MAX as f32)
                        .collect();
                    Ok((out.output_path, samples, sr))
                })();

                let _ = match result {
                    Ok((path, samples, sr)) => tx.send(GuiEvent::Finished {
                        path,
                        samples,
                        sample_rate: sr,
                    }),
                    Err(e) => tx.send(GuiEvent::Failed(format!("Generation failed: {e}"))),
                };
            }
        }
    }
}

impl eframe::App for VoxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Process events ──
        while let Ok(evt) = self.rx.try_recv() {
            match evt {
                GuiEvent::Finished {
                    path,
                    samples,
                    sample_rate,
                } => {
                    let msg = format!(
                        "Finished: {} ({}s, {} samples)",
                        path.display(),
                        samples.len() / sample_rate as usize,
                        samples.len()
                    );
                    self.diagnostics_tab.log.push(msg);
                    self.output_tab.audio_samples = samples;
                    self.output_tab.audio_sample_rate = sample_rate;
                    self.synth_tab.generate_disabled = false;
                    self.active_tab = Tab::Output;
                }
                GuiEvent::Failed(e) => {
                    self.diagnostics_tab.log.push(e.clone());
                    self.synth_tab.generate_disabled = false;
                }
            }
        }

        // ── Top bar ──
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("VoxCPM2 Rust Candle");
                Tab::ui_bar(ui, &mut self.active_tab);
            });
        });

        // ── Content area ──
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.active_tab {
                Tab::Model => {
                    self.model_tab.ui(ui);
                }
                Tab::Synthesis => {
                    self.synth_tab.ui(
                        ui,
                        &self.model_tab.model_dir,
                        &self.model_tab.device_str,
                        &self.cancel_flag,
                    );

                    // Check for pending generate
                    if let Some(req) = self.synth_tab.pending_generate.take() {
                        self.cancel_flag.store(false, Ordering::SeqCst);
                        let _ = self.tx.send(GuiCommand::Generate(req));
                        self.diagnostics_tab
                            .log
                            .push("Generation started...".into());
                    }
                }
                Tab::Cloning => {
                    self.clone_tab.ui(ui);
                }
                Tab::Output => {
                    self.output_tab.ui(ui);
                }
                Tab::Diagnostics => {
                    self.diagnostics_tab.ui(
                        ui,
                        &self.model_tab.model_dir,
                        &self.model_tab.device_str,
                    );
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "VoxCPM2 Rust Candle",
        native_options,
        Box::new(|_cc| Ok(Box::<VoxApp>::default())),
    )
}
