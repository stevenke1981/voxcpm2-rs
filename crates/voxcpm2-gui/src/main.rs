use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use rodio::{buffer::SamplesBuffer, OutputStream, Sink};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};
use voxcpm2_core::{SynthRequest, VoxPipeline};

#[derive(Debug)]
enum GuiCommand {
    Generate(SynthRequest),
}

#[derive(Debug)]
enum GuiEvent {
    Log(String),
    Finished {
        path: PathBuf,
        samples: Vec<f32>,
        sample_rate: u32,
    },
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
    is_playing: Arc<AtomicBool>,
    /// Decoded audio samples (last generated).
    audio_samples: Vec<f32>,
    audio_sample_rate: u32,
    tx: Sender<GuiCommand>,
    rx: Receiver<GuiEvent>,
}

impl Default for VoxApp {
    fn default() -> Self {
        let (cmd_tx, cmd_rx) = unbounded::<GuiCommand>();
        let (evt_tx, evt_rx) = unbounded::<GuiEvent>();
        let playing_flag = Arc::new(AtomicBool::new(false));
        let flag = playing_flag.clone();
        thread::spawn(move || worker_loop(cmd_rx, evt_tx, flag));
        Self {
            model_dir: "models/VoxCPM2".into(),
            text: "你好，這是 VoxCPM2 Rust Candle egui 測試。".into(),
            output: "output/gui_smoke.wav".into(),
            device: "auto".into(),
            cfg: 2.0,
            steps: 10,
            dry_run: true,
            log: vec!["Ready".into()],
            is_playing: playing_flag,
            audio_samples: vec![],
            audio_sample_rate: 48000,
            tx: cmd_tx,
            rx: evt_rx,
        }
    }
}

fn worker_loop(rx: Receiver<GuiCommand>, tx: Sender<GuiEvent>, playing: Arc<AtomicBool>) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            GuiCommand::Generate(req) => {
                let _ = tx.send(GuiEvent::Log("Generating...".into()));
                let result = (|| -> anyhow::Result<(PathBuf, Vec<f32>, u32)> {
                    let mut pipe =
                        VoxPipeline::new(&req.device, req.model_dir.as_deref(), req.dry_run)?;
                    let out = pipe.synthesize(&req)?;
                    let sr = out.sample_rate;

                    // Load WAV for playback
                    let mut reader = hound::WavReader::open(&out.output_path)?;
                    let _spec = reader.spec();
                    let samples: Vec<f32> = reader
                        .samples::<i16>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / i16::MAX as f32)
                        .collect();
                    Ok((out.output_path, samples, sr))
                })();
                playing.store(false, Ordering::SeqCst);
                match result {
                    Ok((path, samples, sr)) => {
                        let _ = tx.send(GuiEvent::Finished {
                            path,
                            samples,
                            sample_rate: sr,
                        });
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
        // ── Process events ──
        while let Ok(evt) = self.rx.try_recv() {
            match evt {
                GuiEvent::Log(s) => self.log.push(s),
                GuiEvent::Finished {
                    path,
                    samples,
                    sample_rate,
                } => {
                    self.log.push(format!(
                        "Finished: {} ({}s)",
                        path.display(),
                        samples.len() / sample_rate as usize
                    ));
                    self.audio_samples = samples;
                    self.audio_sample_rate = sample_rate;
                }
                GuiEvent::Failed(e) => self.log.push(format!("Error: {e}")),
            }
        }

        // ── Top bar ──
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("VoxCPM2 Rust Candle");
                ui.label(format!("device: {}", self.device));
                ui.checkbox(&mut self.dry_run, "dry-run smoke");
            });
        });

        // ── Central panel ──
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

            // ── Audio playback controls ──
            let has_audio = !self.audio_samples.is_empty();
            if has_audio {
                ui.separator();
                ui.heading("Playback");
                ui.horizontal(|ui| {
                    let playing = self.is_playing.load(Ordering::SeqCst);
                    let btn_label = if playing { "⏹ Stop" } else { "▶ Play" };
                    if ui
                        .add_sized([80.0, 30.0], egui::Button::new(btn_label))
                        .clicked()
                    {
                        if playing {
                            self.is_playing.store(false, Ordering::SeqCst);
                        } else {
                            let samples = self.audio_samples.clone();
                            let sr = self.audio_sample_rate;
                            let flag = self.is_playing.clone();
                            flag.store(true, Ordering::SeqCst);
                            thread::spawn(move || {
                                play_audio(&samples, sr, flag);
                            });
                        }
                    }
                    ui.label(format!(
                        "{:.1}s @ {}Hz",
                        self.audio_samples.len() as f64 / self.audio_sample_rate as f64,
                        self.audio_sample_rate
                    ));
                });

                // ── Waveform preview ──
                let total = self.audio_samples.len();
                if total > 0 {
                    let max_f = self
                        .audio_samples
                        .iter()
                        .map(|s| s.abs())
                        .fold(0.0f32, f32::max)
                        .max(1e-8);
                    let painter = ui.painter();
                    let rect = ui.available_rect_before_wrap();
                    let w = (rect.width() - 10.0).min(600.0);
                    let h = 60.0;
                    let origin = egui::pos2(rect.left() + 5.0, rect.top());
                    let mut points: Vec<egui::Pos2> = Vec::with_capacity(512);
                    let step = (total / 512).max(1);
                    for i in (0..total).step_by(step) {
                        let x = origin.x + (i as f32 / total as f32) * w;
                        let y = origin.y + h * 0.5 - (self.audio_samples[i] / max_f) * h * 0.4;
                        points.push(egui::pos2(x, y));
                    }
                    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 180, 255));
                    painter.add(egui::Shape::line(points, stroke));
                    // Reserve space
                    ui.allocate_space(egui::vec2(w, h));
                }
            }

            // ── Diagnostics ──
            ui.separator();
            ui.heading("Diagnostics");
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .show(ui, |ui| {
                    for line in &self.log {
                        ui.label(line);
                    }
                });
        });
    }
}

/// Play audio samples using rodio in a background thread.
fn play_audio(samples: &[f32], sample_rate: u32, playing: Arc<AtomicBool>) {
    // rodio requires the output stream to live for the duration
    let (_stream, stream_handle) = match OutputStream::try_default() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Audio output unavailable: {e}");
            playing.store(false, Ordering::SeqCst);
            return;
        }
    };
    let sink = match Sink::try_new(&stream_handle) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Audio sink error: {e}");
            playing.store(false, Ordering::SeqCst);
            return;
        }
    };

    // Convert f32 samples to rodio source
    let clamped: Vec<f32> = samples.iter().map(|&s| s.clamp(-1.0, 1.0)).collect();
    let source = SamplesBuffer::new(1, sample_rate, clamped);

    sink.append(source);
    // Poll until playback stops or user interrupts
    while !sink.empty() && playing.load(Ordering::SeqCst) {
        thread::sleep(std::time::Duration::from_millis(50));
    }
    sink.stop();
    playing.store(false, Ordering::SeqCst);
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "VoxCPM2 Rust Candle egui",
        native_options,
        Box::new(|_cc| Ok(Box::<VoxApp>::default())),
    )
}
