# egui.md — 桌面 GUI 設計

## 技術選型

- `egui`：immediate mode GUI。
- `eframe`：native desktop runner。
- 背景推理：`std::thread` + `crossbeam-channel`。
- 音訊播放：後續接 `rodio` 或平台播放器。
- 檔案選擇：`rfd`。

## UI Layout

```text
Top bar: Device / Model status / Generate / Cancel
Left panel: tabs
Central panel: active page
Bottom panel: log + progress
```

## Pages

### Model

- Model directory picker。
- Download instructions。
- Inspect button。
- Manifest summary。
- CUDA/Metal/CPU status。

### Synthesis

- Text input。
- Voice Design prompt：`(A young woman, gentle and sweet voice)`。
- cfg slider。
- steps slider。
- seed input。
- output path。
- Generate / Cancel。

### Cloning

- Reference wav picker。
- Prompt text input。
- Style instruction。
- Consent checkbox。

### Output

- Last wav path。
- Play / Stop。
- Export。
- Basic waveform placeholder。

### Diagnostics

- Logs。
- Manifest warnings。
- Build features。
- Benchmark results。

## Worker message protocol

```rust
pub enum GuiCommand {
    InspectModel { model_dir: PathBuf },
    Generate { request: SynthRequest },
    Cancel,
}

pub enum GuiEvent {
    Log(String),
    Progress { done: usize, total: usize },
    Chunk { samples: Vec<f32>, sample_rate: u32 },
    Finished { wav_path: PathBuf },
    Failed { error: String },
}
```

## 不凍結 UI 的規則

- 不在 `App::update` 內載入模型。
- 不在 UI thread 執行 Candle forward。
- 只用 channel 傳狀態。
- 每 frame drain events。
