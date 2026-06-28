use clap::{Parser, Subcommand};
use std::path::PathBuf;
use voxcpm2_core::{full_asset_report, SynthRequest, VoxPipeline};

#[derive(Debug, Parser)]
#[command(name = "voxcpm2", about = "Rust + Candle VoxCPM2 CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Synth {
        #[arg(long)]
        text: String,
        #[arg(long, default_value = "output/output.wav")]
        out: PathBuf,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = 2.5)]
        cfg: f32,
        #[arg(long, default_value_t = 30)]
        steps: usize,
        #[arg(long)]
        max_ar_steps: Option<usize>,
        #[arg(long, default_value_t = 100)]
        seed: u64,
        #[arg(long)]
        voice_design: Option<String>,
        #[arg(long)]
        gain: Option<f32>,
        #[arg(long, default_value = "uniform")]
        t_scheduler: String,
        #[arg(long)]
        latent_norm: Option<f64>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long, default_value_t = true)]
        label_ai_generated: bool,
    },
    /// Voice cloning: synthesize speech that mimics a reference audio.
    Clone {
        #[arg(long)]
        text: String,
        /// Path to reference audio WAV file for voice cloning.
        #[arg(long)]
        ref_audio: PathBuf,
        #[arg(long, default_value = "output/clone.wav")]
        out: PathBuf,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = 2.5)]
        cfg: f32,
        #[arg(long, default_value_t = 30)]
        steps: usize,
        #[arg(long)]
        max_ar_steps: Option<usize>,
        #[arg(long, default_value_t = 100)]
        seed: u64,
        #[arg(long)]
        voice_design: Option<String>,
        #[arg(long)]
        gain: Option<f32>,
        #[arg(long, default_value = "uniform")]
        t_scheduler: String,
        #[arg(long)]
        latent_norm: Option<f64>,
        /// Clone strength (0.0 = text-only, 1.0 = full clone).
        #[arg(long, default_value_t = 1.0)]
        clone_strength: f64,
        #[arg(long)]
        dry_run: bool,
        #[arg(long, default_value_t = true)]
        label_ai_generated: bool,
    },
    Inspect {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        hash: bool,
    },
    Benchmark {
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = 3)]
        repeat: usize,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Command::Synth {
            text,
            out,
            model_dir,
            device,
            cfg,
            steps,
            max_ar_steps,
            seed,
            voice_design,
            gain,
            dry_run,
            label_ai_generated,
            t_scheduler,
            latent_norm,
        } => {
            let mut pipe = VoxPipeline::new(&device, model_dir.as_deref(), dry_run)?;
            let req = SynthRequest {
                text,
                model_dir,
                output_path: out,
                device,
                cfg_value: cfg,
                inference_timesteps: steps,
                max_autoregressive_steps: max_ar_steps,
                seed: Some(seed),
                voice_design,
                post_gain: gain,
                dry_run,
                label_ai_generated,
                t_scheduler,
                latent_norm_scale: latent_norm,
                ref_audio_path: None,
                ref_transcript: None,
                clone_strength: 1.0,
            };
            let result = pipe.synthesize(&req, None)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Clone {
            text,
            ref_audio,
            out,
            model_dir,
            device,
            cfg,
            steps,
            max_ar_steps,
            seed,
            voice_design,
            gain,
            dry_run,
            label_ai_generated,
            t_scheduler,
            latent_norm,
            clone_strength,
        } => {
            let mut pipe = VoxPipeline::new(&device, model_dir.as_deref(), dry_run)?;
            let req = SynthRequest {
                text,
                model_dir,
                output_path: out,
                device,
                cfg_value: cfg,
                inference_timesteps: steps,
                max_autoregressive_steps: max_ar_steps,
                seed: Some(seed),
                voice_design,
                post_gain: gain,
                dry_run,
                label_ai_generated,
                t_scheduler,
                latent_norm_scale: latent_norm,
                ref_audio_path: Some(ref_audio),
                ref_transcript: None,
                clone_strength,
            };
            let result = pipe.synthesize(&req, None)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Inspect { model_dir, hash } => {
            let report = full_asset_report(&model_dir, hash)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.model_ready {
                eprintln!("\n[!] Model not ready. Fix instructions:");
                for fix in &report.fixes {
                    eprintln!("  - {}: {}", fix.file, fix.hint);
                    if let Some(cmd) = &fix.fix_command {
                        eprintln!("    Fix: {cmd}");
                    }
                }
            }
        }
        Command::Benchmark {
            model_dir,
            device,
            repeat,
        } => {
            println!(
                "benchmark placeholder: device={device}, repeat={repeat}, model_dir={:?}",
                model_dir
            );
            println!("real benchmark is enabled after milestones C-G are implemented");
        }
    }
    Ok(())
}
