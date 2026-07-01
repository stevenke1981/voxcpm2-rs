# VoxCPM2 Rust/Candle 改善測試計畫

## 原則

- 所有語音品質修正都要同時看數值與 ASR，不以單一聽感判斷通過。
- Mandarin 基準先用簡體中文與 seed 99；繁體字另作語言偏移測試。
- Clone 測試必須驗證 reference 前處理，避免 reference noise 被 AudioVAE encoder 寫入 speaker conditioning。
- 需要真實模型或 GPU 的測試標為 model gate，不放入一般快速單元測試。

## G0 - 快速健康檢查

```powershell
.\run_with_vs.cmd cargo fmt --all -- --check
.\run_with_vs.cmd cargo test -p voxcpm2-core
.\run_with_vs.cmd cargo test -p voxcpm2-cli
.\run_with_vs.cmd cargo test -p voxcpm2-gui
```

通過條件：

- 無 formatting diff。
- `audio.rs` 的 polish、background gate、clone reference polish、harsh smoother 單元測試通過。
- CLI/GUI crate 維持可編譯。

## G1 - CLI safety gate

未授權 clone 必須失敗：

```powershell
.\run_with_vs.cmd cargo run -p voxcpm2-cli -- clone `
  --model-dir models\VoxCPM2 `
  --ref-audio fixtures\ref.wav `
  --text "这是授权测试。" `
  --out output\should-not-exist.wav
```

授權 clone 才可進入推論：

```powershell
.\run_with_vs.cmd cargo run -p voxcpm2-cli -- clone `
  --model-dir models\VoxCPM2 `
  --ref-audio fixtures\ref.wav `
  --text "这是授权测试。" `
  --i-have-consent `
  --out output\clone-consented.wav `
  --steps 30 --seed 99
```

通過條件：

- 第一個命令非 0 exit code，且不建立輸出 WAV。
- 第二個命令若模型存在，能走到真實 clone；若模型缺失，要輸出明確缺檔診斷。

## G2 - Mandarin synth audio gate

CLI language-risk smoke：

```powershell
.\run_with_vs.cmd cargo run -p voxcpm2-cli --features cpu -- synth `
  --text "你好，這是語音測試。" `
  --out output\traditional_hint_smoke.wav `
  --dry-run
```

通過條件：

- stderr 包含 `Traditional Chinese text detected` 與 `Simplified Chinese text`。
- 使用簡體中文 `你好，这是语音测试。` 時不輸出該語言風險提示。

可直接使用 harness 跑固定 prompt set：

```powershell
.\harness\audio_quality_gate.ps1 `
  -ModelDir models\VoxCPM2 `
  -Device cuda `
  -OutDir output\quality-gate `
  -SkipClone
```

單句手動命令：

```powershell
.\run_with_vs.cmd cargo run -p voxcpm2-cli -- synth `
  --model-dir models\VoxCPM2 `
  --text "这是普通话语音质量回归测试，中间不应该出现刺耳杂音。" `
  --out output\mandarin_seed99.wav `
  --metrics-out output\mandarin_seed99.metrics.json `
  --steps 30 --seed 99 --device cuda
```

通過條件：

- WAV 為 48 kHz mono、finite、non-empty。
- peak <= 0.95，無 NaN/Inf，無 clipping。
- harness 會為每個 `*.metrics.json` 旁輸出 `*.metrics.quality.json`，包含 20ms frame 的
  `high_zcr_frame_count`、`max_zcr`、`max_frame_rms` 與對應 threshold。
- quiet_rms 不高於前一個 accepted baseline。
- high-ZCR / 4-8kHz burst proxy frame 數不高於前一個 accepted baseline。
- ASR transcript 與原文高度一致；至少不得漏掉主要詞：「普通话」、「语音质量」、「刺耳杂音」。

## G3 - Clone reference noise gate

測試組：

- clean reference。
- background-noisy reference。
- long reference (>30s)。

快速單元 fixture gate：

```powershell
.\run_with_vs.cmd cargo test -p voxcpm2-core clone_reference_fixture_matrix --features cpu
```

通過條件：

- clean reference 經 `polish_clone_reference_audio` 後 speech RMS 保留至少 80%。
- noisy reference 的 `quiet_rms_after` 明顯低於 `quiet_rms_before`，speech RMS 保留至少 65%。
- long reference 會被裁切到 30 秒，並回傳 `clone_reference_trim` 診斷欄位。

可直接使用 harness 跑 clone gate：

```powershell
.\harness\audio_quality_gate.ps1 `
  -ModelDir models\VoxCPM2 `
  -Device cuda `
  -RefAudio fixtures\clone\noisy_ref.wav `
  -OutDir output\quality-gate
```

單句手動命令：

```powershell
.\run_with_vs.cmd cargo run -p voxcpm2-cli -- clone `
  --model-dir models\VoxCPM2 `
  --ref-audio fixtures\clone\noisy_ref.wav `
  --text "这是声音复制的噪声回归测试。" `
  --i-have-consent `
  --out output\clone_noisy_ref_seed99.wav `
  --metrics-out output\clone_noisy_ref_seed99.metrics.json `
  --steps 30 --seed 99 --device cuda
```

通過條件：

- clone reference polish report 顯示 quiet_rms_after < quiet_rms_before。
- clone metrics 若 reference 超過 30 秒，`clone_reference_trim` 顯示原始長度、裁切長度與 sample rate。
- 背景 gate threshold 在合理範圍內，不把 reference 語音削成靜音。
- long reference 顯示 trim log，且不造成 CUDA OOM。
- ASR 可辨識目標文字。

## G4 - Clone sequence parity gate

對照 Python 或 C++ fixture 建立 synthetic reference：

- 16001 samples 220 Hz sine。
- right padding for reference-only。
- left padding for prompt continuation。
- combined mode queue order: reference patches then prompt patches。

通過條件：

- combined token ids 長度、ref start/end 位置、text/audio mask 與 fixture 一致。
- patch count 與 `patch_size = 4` 對齊。
- `clone_strength = 0.0` 接近 text-only，`clone_strength = 1.0` 使用完整 feat embed。

## G5 - Backend/performance matrix

最低矩陣：

| Case | Device | Test |
| --- | --- | --- |
| cpu-synth | CPU | short Mandarin synth |
| cuda-synth | CUDA | short Mandarin synth |
| cuda-clone | CUDA | authorized clone smoke |

每列紀錄：

- command。
- commit。
- model hash 或 model directory manifest。
- wall time。
- audio seconds。
- RTF。
- peak working set 或 CUDA memory。
- WAV metrics。
- ASR transcript。

通過條件：

- 每列皆 finite、non-empty、無 clipping。
- CUDA synth/clone 不比前一 accepted baseline 明顯退化。
- 失敗時保留 stderr、metrics、輸出路徑或缺檔診斷。

可用 baseline writer 產生 release evidence：

```powershell
.\harness\write_acceptance_baseline.ps1 `
  -ModelDir models\VoxCPM2 `
  -Device cuda `
  -RefAudio fixtures\clone\noisy_ref.wav `
  -OutDir output\accepted-baseline `
  -RunAsr `
  -UpdateAcceptanceReport
```

快速驗證 baseline writer 本身：

```powershell
.\harness\write_acceptance_baseline.ps1 `
  -Device cpu `
  -DryRun `
  -SkipClone `
  -OutDir output\accepted-baseline-dryrun
```

通過條件：

- `baseline_manifest.json` 包含 commit、command、device、seed、每個 case 的 WAV/metrics/quality path。
- `baseline_report.md` 可讀取每個 case 的 peak、high-ZCR frame 數與 ASR 狀態。
- 真實模型 accepted baseline 必須使用 `-RunAsr`；若未跑 ASR，狀態只能是 `needs_asr`。
- dry-run 只能證明 harness 可執行，不能標成真實語音 accepted baseline。

## G6 - GUI smoke

手動或 Playwright/Windows UI harness：

- Model tab 能選 model dir。
- Synth tab 產生 seed 99 Mandarin WAV。
- Clone tab 未勾 consent 時不可送出。
- 勾 consent 並指定 reference 後可送出。
- Synth/Clone tab 送出時會在輸出 WAV 旁建立 `*.metrics.json`。
- Cancel 能停止長時間推論並回報取消位置。

通過條件：

- GUI 與 CLI 使用同一套安全與 audio quality gate。
- 錯誤訊息不吞掉 shape mismatch、缺檔、CUDA、AudioVAE 診斷。

## G7 - Release gate

```powershell
.\run_with_vs.cmd cargo build --release --features cuda
.\harness\smoke_cli.ps1
```

通過條件：

- release build 成功。
- package 不含模型權重、WAV、fixtures、debug tensors、ASR artifacts。
- `README.md`、`docs/todos.md`、`acceptance_report.md` 的狀態與實測一致。
