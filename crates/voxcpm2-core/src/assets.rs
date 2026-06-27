use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetStatus {
    pub path: PathBuf,
    pub exists: bool,
    pub bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsManifest {
    pub path: PathBuf,
    pub exists: bool,
    pub num_tensors: Option<usize>,
    pub total_params: Option<u64>,
    pub dtype_counts: Option<BTreeMap<String, usize>>,
    pub tensor_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixInstruction {
    pub file: String,
    pub missing: bool,
    pub fix_command: Option<String>,
    pub hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetReport {
    pub files: Vec<AssetStatus>,
    pub manifest: SafetensorsManifest,
    pub audiovae_manifest: Option<SafetensorsManifest>,
    pub fixes: Vec<FixInstruction>,
    pub model_ready: bool,
}

pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
];

pub fn inspect_required(
    model_dir: impl AsRef<Path>,
    with_hash: bool,
) -> anyhow::Result<Vec<AssetStatus>> {
    let model_dir = model_dir.as_ref();
    let mut out = Vec::new();
    for name in REQUIRED_FILES
        .iter()
        .copied()
        .chain(["audiovae.safetensors", "audiovae.pth"])
    {
        let path = model_dir.join(name);
        let exists = path.exists();
        let bytes = if exists {
            Some(path.metadata()?.len())
        } else {
            None
        };
        let sha256 = if exists && with_hash {
            Some(sha256_file(&path)?)
        } else {
            None
        };
        out.push(AssetStatus {
            path,
            exists,
            bytes,
            sha256,
        });
    }
    Ok(out)
}

pub fn inspect_safetensors(path: &Path) -> SafetensorsManifest {
    if !path.exists() {
        return SafetensorsManifest {
            path: path.to_path_buf(),
            exists: false,
            num_tensors: None,
            total_params: None,
            dtype_counts: None,
            tensor_names: None,
        };
    }
    match read_safetensors_metadata(path) {
        Ok((num_tensors, total_params, dtype_counts, tensor_names)) => SafetensorsManifest {
            path: path.to_path_buf(),
            exists: true,
            num_tensors: Some(num_tensors),
            total_params: Some(total_params),
            dtype_counts: Some(dtype_counts),
            tensor_names: Some(tensor_names),
        },
        Err(e) => SafetensorsManifest {
            path: path.to_path_buf(),
            exists: true,
            num_tensors: None,
            total_params: None,
            dtype_counts: None,
            tensor_names: Some(vec![format!("error reading safetensors: {e}")]),
        },
    }
}

type MetaResult = (usize, u64, BTreeMap<String, usize>, Vec<String>);

fn read_safetensors_metadata(path: &Path) -> anyhow::Result<MetaResult> {
    let mut file = File::open(path)?;
    let mut header_len_buf = [0u8; 8];
    file.read_exact(&mut header_len_buf)?;
    let header_len = u64::from_le_bytes(header_len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)?;

    let mut dtype_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_params: u64 = 0;
    let mut tensor_names = Vec::new();

    for (name, info) in &header {
        if name == "__metadata__" {
            continue;
        }
        tensor_names.push(name.clone());
        if let Some(dtype) = info.get("dtype").and_then(|v| v.as_str()) {
            *dtype_counts.entry(dtype.to_string()).or_insert(0) += 1;
        }
        if let Some(shape) = info.get("shape").and_then(|v| v.as_array()) {
            let n: u64 = shape.iter().filter_map(|v| v.as_u64()).product();
            total_params += n;
        }
    }
    tensor_names.sort();

    Ok((tensor_names.len(), total_params, dtype_counts, tensor_names))
}

pub fn generate_fix_instructions(
    model_dir: impl AsRef<Path>,
    status: &[AssetStatus],
) -> Vec<FixInstruction> {
    let model_dir = model_dir.as_ref();
    let mut fixes = Vec::new();

    for s in status {
        if s.exists {
            continue;
        }
        let fname = s
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let (fix_command, hint) = match fname.as_str() {
            "config.json" => (
                Some(
                    "Download HF model snapshot manually or run: scripts/download_model.py".into(),
                ),
                "Required for model architecture definition".into(),
            ),
            "tokenizer.json" => (
                Some(
                    "Download HF model snapshot manually or run: scripts/download_model.py".into(),
                ),
                "Required for text tokenization".into(),
            ),
            "tokenizer_config.json" => (
                Some(
                    "Download HF model snapshot manually or run: scripts/download_model.py".into(),
                ),
                "Required for special tokens & chat template".into(),
            ),
            "special_tokens_map.json" => (
                Some(
                    "Download HF model snapshot manually or run: scripts/download_model.py".into(),
                ),
                "Required for special token mapping".into(),
            ),
            "model.safetensors" => (
                Some(
                    "Download HF model snapshot manually or run: scripts/download_model.py".into(),
                ),
                "Core model weights (~2.5GB)".into(),
            ),
            "audiovae.safetensors" => {
                let pth_path = model_dir.join("audiovae.pth");
                if pth_path.exists() {
                    (
                        Some("python scripts/convert_audiovae_pth_to_safetensors.py --input audiovae.pth --output audiovae.safetensors".into()),
                        "audiovae.pth exists; convert to safetensors".into(),
                    )
                } else {
                    (
                        Some("Download HF model snapshot (includes audiovae.pth) then run conversion script".into()),
                        "AudioVAE decoder weights needed for waveform generation".into(),
                    )
                }
            }
            "audiovae.pth" => {
                let st_path = model_dir.join("audiovae.safetensors");
                if st_path.exists() {
                    (
                        None,
                        "audiovae.safetensors already exists; .pth not needed".into(),
                    )
                } else {
                    (
                        Some(
                            "Download HF model snapshot manually or run: scripts/download_model.py"
                                .into(),
                        ),
                        "AudioVAE weights in PyTorch format; convert to safetensors for Rust"
                            .into(),
                    )
                }
            }
            _ => (None, "Unknown file".into()),
        };
        fixes.push(FixInstruction {
            file: fname,
            missing: true,
            fix_command,
            hint,
        });
    }
    fixes
}

pub fn full_asset_report(
    model_dir: impl AsRef<Path>,
    with_hash: bool,
) -> anyhow::Result<AssetReport> {
    let model_dir = model_dir.as_ref();
    let files = inspect_required(model_dir, with_hash)?;
    let manifest = inspect_safetensors(&model_dir.join("model.safetensors"));
    let audiovae_manifest = {
        let st = model_dir.join("audiovae.safetensors");
        if st.exists() {
            Some(inspect_safetensors(&st))
        } else {
            None
        }
    };
    let fixes = generate_fix_instructions(model_dir, &files);
    let model_ready = assert_minimum_assets(model_dir).is_ok();
    Ok(AssetReport {
        files,
        manifest,
        audiovae_manifest,
        fixes,
        model_ready,
    })
}

pub fn assert_minimum_assets(model_dir: impl AsRef<Path>) -> anyhow::Result<()> {
    let model_dir = model_dir.as_ref();
    for name in REQUIRED_FILES {
        let path = model_dir.join(name);
        if !path.exists() {
            anyhow::bail!("missing required model file: {}", path.display());
        }
    }
    let st = model_dir.join("audiovae.safetensors");
    let pth = model_dir.join("audiovae.pth");
    if !st.exists() && !pth.exists() {
        anyhow::bail!("missing AudioVAE weights: expected audiovae.safetensors or audiovae.pth (run `python scripts/convert_audiovae_pth_to_safetensors.py` if you have audiovae.pth)");
    }
    Ok(())
}

pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_fix_instructions_all_missing() {
        let dir = PathBuf::from("/nonexistent/models");
        // create fake AssetStatus entries all missing
        let files = vec![
            AssetStatus {
                path: dir.join("config.json"),
                exists: false,
                bytes: None,
                sha256: None,
            },
            AssetStatus {
                path: dir.join("tokenizer.json"),
                exists: false,
                bytes: None,
                sha256: None,
            },
        ];
        let fixes = generate_fix_instructions(&dir, &files);
        assert_eq!(fixes.len(), 2);
        assert!(fixes[0].fix_command.is_some());
        assert!(fixes[1].missing);
    }

    #[test]
    fn full_report_on_nonexistent_dir() {
        let report = full_asset_report(PathBuf::from("/nonexistent/models"), false).unwrap();
        assert!(!report.model_ready);
        assert!(report.fixes.len() >= 5);
    }
}
