# architecture.md — 架構設計

## Workspace

```text
voxcpm2-rust-candle-egui
├─ crates/voxcpm2-core
├─ crates/voxcpm2-cli
└─ crates/voxcpm2-gui
```

## 核心資料流

```text
text / prompt / reference wav
  ↓
TextNormalizer + VoxCPM2Tokenizer
  ↓
PromptBuilder：BOS / special tokens / voice design / clone fields
  ↓
ModelInputs：input_ids, optional acoustic prompt, masks
  ↓
LocEnc
  ↓
TSLM backbone
  ↓
RALM residual acoustic LM
  ↓
LocDiT diffusion transformer + CFG + Euler scheduler
  ↓
AudioVAE V2 decode
  ↓
PCM f32 → WAV 48kHz
```

## 模組

| Rust module | 責任 |
|---|---|
| `config` | 解析 `config.json` 與本地 TOML |
| `device` | CPU/CUDA/Metal/MKL 選擇與診斷 |
| `assets` | 檔案檢查、manifest、sha256 |
| `tokenizer` | tokenizers crate + VoxCPM2 custom 行為 |
| `layers` | RMSNorm、Linear、Attention、RoPE、MLP |
| `tslm` | 28-layer LM backbone |
| `ralm` | residual acoustic LM |
| `locenc` | local acoustic encoder |
| `locdit` | diffusion transformer |
| `scheduler` | flow matching / Euler solver |
| `audiovae` | AudioVAE V2 decoder |
| `pipeline` | generate / streaming generate |
| `audio` | wav read/write、sanity check |
| `diagnostics` | benchmark、memory report、shape dump |

## GUI Threading

```text
egui main thread
  ├─ UI state
  ├─ command channel → worker thread
  └─ result channel ← worker thread

worker thread
  ├─ model loader
  ├─ synth pipeline
  ├─ streaming chunks
  └─ cancel token
```

GUI 不可在 main UI frame 中執行長時間推理，否則視窗會 freeze。

## Error taxonomy

| Error | 說明 | 修復 |
|---|---|---|
| MissingAsset | 缺模型檔案 | 執行 download script |
| UnsafePth | 偵測到 `.pth` 未轉換 | 執行 convert script |
| ShapeMismatch | 張量 shape 不符 | 檢查 mapping / config |
| DeviceUnavailable | CUDA/Metal 不可用 | 切 CPU 或修 driver |
| Oom | VRAM/RAM 不足 | 降 steps、短文本、CPU offload |
| AudioInvalid | 輸出音訊異常 | 檢查 AudioVAE / scheduler |
