# spec.md — VoxCPM2 Rust + Candle + egui 規格

## 1. 專案目標

建立一個 Rust 原生 VoxCPM2 推理專案：

- 使用 Candle 執行張量運算、模型載入與 GPU/CPU 後端管理。
- 提供 CLI 與 egui 桌面 GUI。
- 支援 Hugging Face 模型下載、權重檢查、`audiovae.pth` 轉 safetensors。
- 分階段實作 VoxCPM2 推理管線，從 smoke audio 到真實 TTS。

## 2. 非目標

初版不做：

- 不訓練模型。
- 不直接在 Rust 內解析 PyTorch pickle `.pth`。
- 不承諾與官方 Python 100% bit-identical；目標是音訊品質與張量 shape 對齊，再逐步做誤差對齊。
- 不實作非法聲音冒充、欺詐或未授權克隆流程。

## 3. 功能需求

### F1. 模型資產管理

- 可指定 `--model-dir models/VoxCPM2`。
- 檢查以下檔案：
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `model.safetensors`
  - `audiovae.safetensors` 或 `audiovae.pth`
- 若只存在 `audiovae.pth`，提示先執行轉換腳本。
- 產生 `model_manifest.json`，記錄檔案大小、sha256、張量數量、dtype 分布。

### F2. CLI

```bash
voxcpm2 synth --text "..." --out output.wav --model-dir models/VoxCPM2 --device auto --cfg 2.0 --steps 10
voxcpm2 inspect --model-dir models/VoxCPM2
voxcpm2 benchmark --model-dir models/VoxCPM2 --device cuda --repeat 3
voxcpm2 convert-audiovae --input audiovae.pth --output audiovae.safetensors
```

### F3. egui GUI

GUI 頁面：

1. Model：模型資料夾、下載狀態、manifest、device 狀態。
2. Synthesis：文字輸入、Voice Design 描述、cfg、steps、seed、輸出路徑。
3. Cloning：reference wav、reference transcript、style control。
4. Output：音訊播放、波形預覽、匯出 wav。
5. Diagnostics：GPU/CPU、VRAM/RAM 估算、log、錯誤說明。

### F4. GPU 加速

- `cpu` feature：預設 CPU backend。
- `cuda` feature：NVIDIA GPU，Candle CUDA backend。
- `metal` feature：Apple Silicon Metal backend。
- `mkl` feature：x86 CPU 加速。
- `auto` device 選擇：CUDA → Metal → CPU。
- CLI 與 GUI 都顯示實際 device。

### F5. 推理管線

Rust module 對應：

```text
Tokenizer / text normalizer
  ↓
LocEnc / local acoustic encoder
  ↓
TSLM / MiniCPM-like LM backbone
  ↓
RALM / residual acoustic LM
  ↓
LocDiT / diffusion transformer + flow matching scheduler
  ↓
AudioVAE V2 decoder
  ↓
48kHz wav writer
```

### F6. Streaming

- CLI 支援 `--stream chunks/` 輸出 chunk wav。
- GUI 使用 channel 接收生成 chunk，更新進度條與波形。
- 實作採分階段：先 fake chunks，再真實 LocDiT chunk decode。

## 4. 非功能需求

- Windows 優先，其次 Linux/macOS。
- Rust stable。
- 大模型檔案不進 git。
- 錯誤訊息需包含：缺少檔案、shape mismatch、device 不支援、CUDA 初始化失敗、VRAM 不足。
- 測試需能在沒有模型權重時跑 smoke path。
- 真實模型測試使用 `VOXCPM2_MODEL_DIR` 環境變數啟用。

## 5. 安全需求

- GUI 明示「AI-generated voice」標籤。
- 克隆頁面加入授權聲明 checkbox。
- CLI 克隆模式需加 `--i-have-rights-to-this-voice`。
- 不提供繞過浮水印、冒充真人、詐騙劇本相關功能。

## 6. 驗收標準

| Gate | 條件 |
|---|---|
| G0 scaffold | `cargo test --workspace --features cpu` 成功 |
| G1 model inspect | 可讀 config/tokenizer/model.safetensors manifest |
| G2 tokenizer parity | 20 條中英日韓樣本 token IDs 與 Python 對齊 |
| G3 tensor load | 全部 safetensors 張量可依 manifest 映射 |
| G4 submodule forward | LocEnc/TSLM/RALM/LocDiT/AudioVAE shape tests pass |
| G5 audio smoke | 產生 48kHz wav，無 NaN/Inf/爆音 peak |
| G6 quality parity | 與官方 Python 相同輸入對比 MOS/相似度可接受 |
| G7 GUI | egui 可選模型、生成、播放/匯出、顯示 log |
| G8 GPU | CUDA/Metal 至少一種通過 benchmark |
