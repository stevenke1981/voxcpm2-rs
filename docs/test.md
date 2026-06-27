# test.md — 測試與驗收 Harness

## 1. 測試分層

| 層級 | 目的 | 無模型可跑 | 需模型 |
|---|---|---:|---:|
| Unit | config/device/audio/tokenizer 基本函式 | ✅ | ❌ |
| Smoke | CLI/GUI 基本流程、產生測試 wav | ✅ | ❌ |
| Manifest | 檢查模型檔案與 safetensors | ❌ | ✅ |
| Parity | 與官方 Python token/hidden/audio 對齊 | ❌ | ✅ |
| GPU | CUDA/Metal 初始化與 benchmark | ❌ | ✅ |
| GUI | egui 操作流程 | ✅ | 部分 |

## 2. 基本命令

### Windows

```powershell
./harness/run_all.ps1
./harness/gpu_probe.ps1 -Device cuda
./harness/smoke_cli.ps1
```

### Linux/macOS

```bash
bash harness/run_all.sh
```

## 3. 無模型 smoke gate

```bash
cargo fmt --all -- --check
cargo clippy --workspace --features cpu -- -D warnings
cargo test --workspace --features cpu
cargo run -p voxcpm2-cli --features cpu -- synth --text "smoke" --out output/smoke.wav --dry-run
```

驗收：

- `output/smoke.wav` 存在。
- sample rate = 48000。
- duration > 0.1s。
- 無 panic。

## 4. 模型 manifest gate

```bash
python scripts/inspect_safetensors.py --model-dir models/VoxCPM2 --out models/VoxCPM2/model_manifest.json
cargo run -p voxcpm2-cli --features cpu -- inspect --model-dir models/VoxCPM2
```

驗收：

- `model.safetensors` 可讀。
- dtype 統計包含 BF16 或 F16/F32 fallback。
- 必要 config 欄位存在。

## 5. Tokenizer parity gate

準備：用官方 Python 產生 token IDs JSON。

```bash
python tests/golden/make_tokenizer_golden.py --model-dir models/VoxCPM2 --out tests/golden/tokenizer_cases.json
cargo test -p voxcpm2-core tokenizer_parity --features cpu -- --ignored
```

驗收：

- 中英日韓與 Voice Design 格式 token IDs 完全一致。

## 6. Audio sanity gate

```bash
cargo run -p voxcpm2-cli --features cuda -- synth --model-dir models/VoxCPM2 --text "你好，這是測試。" --out output/real.wav
python tests/golden/check_wav.py output/real.wav --sample-rate 48000
```

驗收：

- sample rate = 48000。
- amplitude peak <= 1.0。
- 無 NaN/Inf。
- 頻譜不是全高頻噪音。

## 7. GPU benchmark gate

```bash
cargo run -p voxcpm2-cli --no-default-features --features cuda -- benchmark --model-dir models/VoxCPM2 --repeat 3
```

輸出格式：

```json
{
  "device": "cuda:0",
  "text_chars": 32,
  "steps": 10,
  "audio_seconds": 3.2,
  "wall_seconds": 4.0,
  "rtf": 1.25,
  "peak_vram_mb": 7900
}
```

## 8. GUI gate

1. 啟動 `cargo run -p voxcpm2-gui --features cpu`。
2. 選模型資料夾。
3. 點 Inspect。
4. 輸入文字。
5. 點 Generate。
6. 播放或匯出 wav。

驗收：GUI 不 freeze；生成期間可取消；錯誤顯示可讀修復建議。
