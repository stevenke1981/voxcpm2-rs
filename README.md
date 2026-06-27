# voxcpm2-rust-candle-egui 開發包

目標：將 `openbmb/VoxCPM2` 以 Rust 重新實作推理管線，使用 Hugging Face Candle 管理張量、模型載入、GPU/CPU 後端，並提供 CLI + egui 桌面介面。

> 這是一個「可交給 OpenCode/Codex 開始實作」的規劃與 harness 包，不包含 4.58GB 模型權重。初始 Rust 程式骨架以 smoke path 產生測試音檔，方便先驗證 CLI / GUI / harness；實際模型層依 `docs/todos.md` 分階段替換。

## 已納入需求

- Rust workspace 架構
- Candle backend：CPU / CUDA / Metal feature gate
- 模型下載、轉換、驗證流程
- AudioVAE `.pth` 轉 `.safetensors` 策略
- egui / eframe 桌面 GUI
- CLI 生成指令
- 測試 harness：PowerShell + Bash
- CI workflow 範本
- OpenCode/Codex 交付用文件：`plan.md`、`spec.md`、`todos.md`、`test.md`、`final.md`

## 目錄

```text
.
├─ docs/
│  ├─ plan.md
│  ├─ spec.md
│  ├─ todos.md
│  ├─ test.md
│  ├─ final.md
│  ├─ architecture.md
│  ├─ conversion.md
│  ├─ gpu.md
│  ├─ egui.md
│  ├─ harness.md
│  └─ model_notes.md
├─ crates/
│  ├─ voxcpm2-core/      # Candle / pipeline / config / audio
│  ├─ voxcpm2-cli/       # CLI
│  └─ voxcpm2-gui/       # egui desktop app
├─ scripts/
│  ├─ download_model.py
│  ├─ convert_audiovae_pth_to_safetensors.py
│  └─ inspect_safetensors.py
├─ harness/
│  ├─ run_all.ps1
│  ├─ run_all.sh
│  ├─ gpu_probe.ps1
│  └─ smoke_cli.ps1
└─ configs/
   └─ voxcpm2.local.example.toml
```

## 快速開始

### 1. 只測骨架

```powershell
cd voxcpm2-rust-candle-egui
cargo test --workspace --features cpu
cargo run -p voxcpm2-cli --features cpu -- synth --text "你好，這是 VoxCPM2 Rust Candle 測試。" --out output/smoke.wav --dry-run
cargo run -p voxcpm2-gui --features cpu
```

### 2. 下載模型

```powershell
python scripts/download_model.py --repo openbmb/VoxCPM2 --out models/VoxCPM2
python scripts/inspect_safetensors.py --model-dir models/VoxCPM2
```

### 3. 轉換 AudioVAE

```powershell
python scripts/convert_audiovae_pth_to_safetensors.py --input models/VoxCPM2/audiovae.pth --output models/VoxCPM2/audiovae.safetensors
```

### 4. GPU build 範例

```powershell
# NVIDIA CUDA
cargo run -p voxcpm2-cli --no-default-features --features cuda -- synth --text "GPU 測試" --out output/gpu.wav

# Apple Metal
cargo run -p voxcpm2-cli --no-default-features --features metal -- synth --text "Metal 測試" --out output/metal.wav
```

## 重要限制

1. VoxCPM2 不是單純 LLaMA 架構；必須重建 LocEnc → TSLM → RALM → LocDiT → AudioVAE V2。
2. 官方權重含 `model.safetensors` 與 `audiovae.pth`；Rust/Candle 可直接處理 safetensors，但 PyTorch pickle `.pth` 建議先離線轉成 safetensors。
3. 真正可用的 TTS 需要完成張量名稱對應、每層 forward、Flow Matching / Euler scheduler、AudioVAE decoder。
4. 聲音克隆能力涉及濫用風險；GUI 與 CLI 預設加入 `--label-ai-generated` 與安全提示。
