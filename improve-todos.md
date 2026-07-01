# VoxCPM2 Rust/Candle 改善待辦

## 背景

本計畫以目前 `E:\voxcpm2_rust_candle_pack` 的 CBM 架構檢視為基礎，並參考
`stevenke1981/voxcpm-cpp` shallow clone `896f595`。目前 Rust 版本已具備
CUDA 文字生成語音、voice clone、AudioVAE encoder/decoder、輸出端 speech polish、
clone reference polish、egui worker 與 model cache；接下來的重點不是再大改聲音濾波，
而是把「可聽、可複製」升級成「可測、可回歸、可發布」。

## P0 - 安全與產品契約

- [x] **強制 clone consent CLI gate**
  - 目前 `configs/voxcpm2.local.example.toml` 已有 `require_voice_clone_consent = true`，
    且 `crates/voxcpm2-cli/src/main.rs` 的 `Clone` command 已強制 `--i-have-consent`。
  - 對齊 `voxcpm-cpp`：任何 reference/prompt audio clone 都必須明確 consent。
  - 驗收：未帶 consent 的 clone 以非 0 exit code 結束，錯誤訊息說明必須取得合法授權；
    帶 consent 且其他參數正確時維持現有行為。
  - 狀態：已補 CLI gate、單元測試、README 範例；GUI consent gate 仍列在 P2。

- [ ] **避免隱藏生成語音身分**
  - 保留 `label_ai_generated` 預設 true。
  - 若未來允許關閉標示，CLI/GUI 必須清楚揭露風險，不能把關閉標示當作一般品質選項。
  - 驗收：README、CLI help、GUI clone tab 的文案一致。

## P1 - 文字與語言輸入品質 gate

- [x] **建立 Mandarin regression prompt set**
  - 對 Mandarin 測試使用簡體中文與 seed 99 作為第一基準，避免繁體字觸發廣東話傾向。
  - 至少包含：短句、長句、含擦音句、含停頓句。
  - 驗收：每個輸出都保存 WAV metrics、ASR transcript、command、seed、model hash。
  - 狀態：已新增 `tests/golden/mandarin_regression_prompts.json` 與 `harness/audio_quality_gate.ps1`；
    目前可產生 WAV/metrics 並做結構檢查，ASR transcript 比對仍是下一層 gate。

- [x] **保留繁體中文風險提示並加測試**
  - 目前 `pipeline.rs` 已有 Traditional Chinese hint detection。
  - 增加 CLI smoke，確認偵測時會提示「若要 Mandarin 請用簡體中文」。
  - 驗收：提示只對繁體提示觸發，簡體 Mandarin 基準不觸發。
  - 狀態：已將語言風險提示移到 `synthesize()` 共用入口，dry-run CLI smoke 也會提示；
    已補單元測試覆蓋繁體文字、voice design 與簡體 Mandarin 不誤觸。

## P1 - 音訊品質回歸

- [x] **把 speech polish 量測輸出改成機器可讀報告**
  - `audio.rs` 已回傳 `AudioPolishReport`；`synth`/`clone` CLI 已新增 `--metrics-out` 保存 JSON。
  - 驗收：每次 synth/clone 都可取得 dc、peak、quiet_rms、bg_gate、harsh_frames。
  - 狀態：已補 dry-run metrics 測試；真實模型 gate 需在 G2/G3 輸出 accepted baseline。

- [ ] **針對中段雜音建立 fixture**
  - 以 `mandarin_bg_gate_seed99.wav` 症狀為模式，建立 high-ZCR / 4-8kHz burst 檢查。
  - 驗收：`apply_harsh_midband_smoother` 單元測試之外，至少一個端到端 sample gate
    能檢出中段刺耳 frame，並確認修正後 ASR 不退化。
  - 狀態：已在 `harness/audio_quality_gate.ps1` 新增 20ms frame-level high-ZCR/harsh-frame
    sidecar JSON；仍需用真實模型 sample 與 ASR transcript 建立 accepted baseline。

- [x] **clone reference 前處理回歸**
  - 目前 clone reference 已在 AudioVAE encoder 前做 `polish_clone_reference_audio`。
  - 補 clone reference fixture：乾淨 reference、帶背景噪 reference、過長 reference。
  - 驗收：帶背景噪 reference 的 quiet_rms 降低，reference 不因 gate 被削到不可辨識；
    超過 30 秒 reference 有明確 trim log。
  - 狀態：已新增 clean/noisy/long synthetic fixture matrix；`SynthMetricsReport` 會輸出
    `clone_reference_trim`，真實模型 gate 仍需補 ASR transcript baseline。

- [x] **建立 multi-seed / CFG / latent normalization quality sweep**
  - 新增 `harness/quality_sweep.ps1`，可掃 seed、CFG、latent norm 與 scheduler。
  - 驗收：輸出 CSV/JSON/Markdown summary，分數同時納入 ASR required-term、normalized text similarity、
    high-ZCR frame、quiet RMS、clone reference floor 與 near-clipping penalty。
  - 狀態：已完成 harness，並已用 CUDA + ASR 跑 seed 102 targeted probe；`cfg=2.5`、latent default、
    uniform scheduler 目前在 Mandarin short/midburst 兩句最佳。下一步是把該候選放進完整矩陣，補
    clone 與 fricatives 後再決定是否更新 accepted baseline。

## P1 - Voice clone parity 與模式完整性

- [ ] **補齊 clone 模式矩陣**
  - 參考 `voxcpm-cpp` 的 reference-only、prompt-only continuation、combined 三種模式。
  - Rust CLI 已對齊 `voxcpm-cpp` clone 輸入形態：`--ref-audio`、`--prompt-audio --prompt-text`、
    以及 reference + prompt combined；pipeline 會將 `prompt_text + target_text` 先拼成同一段
    UTF-8 再 tokenizer，避免 BPE 邊界偏移。
  - 驗收：CLI/GUI 至少定義 roadmap 與錯誤訊息；實作後三種模式都要有 smoke。
  - 狀態：CLI/pipeline 基礎已完成並有單元測試；GUI 仍只暴露 reference-only UI，prompt/combined
    需要下一輪補 UI 控制與真實模型 smoke。

- [x] **clone sequence/token mask parity**
  - Rust clone sequence 已對齊官方 Python：
    reference-only = `[ref_audio_start, ref_patches, ref_audio_end, text+audio_start]`；
    prompt-only = `[prompt_text+target_text+audio_start, prompt_patches]`；
    combined = `[ref_prefix, prompt_text+target_text+audio_start, prompt_patches]`。
  - 對照 Python/C++ fixture，檢查 reference padding 方向、prompt padding 方向、first generation
    position、text/audio mask。
  - 驗收：固定 synthetic audio fixture 的 combined ids、mask 長度、patch 數一致。
  - 狀態：已補 reference right-padding、prompt left-padding、prompt 最後一個 latent patch 作為
    CFM initial condition、audio patch token id=`0`、AR max_len 使用 target text token 長度；
    仍需改善 prompt-only/combined clone 的 ASR 細節與 stop head 穩定性。

- [ ] **prompt/combined clone quality follow-up**
  - 本輪真實 CUDA smoke：
    - prompt-only: `output/alignment_prompt_only_seed102.wav`，ASR 可抓到主要語意但「这是接续生成」
      被辨成近似錯詞。
    - combined: `output/alignment_combined_seed102_v2.wav`，已由 target-length cap 限制到 30.4s，
      但 faster-whisper large-v3-turbo CUDA 對該檔 crash。
  - 下一步：針對 prompt/combined clone 做 seed/cfg/stop threshold 小矩陣，並切 5-10 秒窗做 ASR
    定位，確認是 stop head、latent 分佈還是 ASR engine 對該音訊不穩。

## P2 - 效能與可發布性

- [ ] **建立 backend matrix**
  - 參考 `voxcpm-cpp` 的 CPU/CUDA x F16/Q8 report；Rust 先以 CPU/CUDA x current safetensors 為主。
  - 驗收：短 TTS、clone smoke 產生 finite non-empty WAV，紀錄 wall time、RTF、peak memory。

- [ ] **GUI 端維持同一組 quality/safety gate**
  - GUI clone tab 必須同 CLI 一樣要求 consent，並顯示生成語音標示狀態。
  - Background worker、cancel、model cache 不可因新增 gate regression。
  - 驗收：GUI smoke 能完成 synth、clone、cancel，錯誤訊息不吞掉底層 shape/audio diagnostics。
  - 狀態：已補 GUI clone consent checkbox 與 synth/clone metrics sidecar path；仍需人工或自動 GUI smoke。

- [ ] **Release package hygiene**
  - 參考 C++ repo 的 `scripts/build-release.ps1`：隔離 build、跑 gate、打包不含模型/WAV/fixture。
  - Rust 版本建立 Windows portable zip 流程。
  - 驗收：zip 不含 `models/`、`.safetensors`、`.wav`、debug dump、ASR artifacts。

## P2 - 文件與追蹤

- [ ] **更新 `docs/todos.md` 的 Milestone I/J**
  - 將本文件 P0/P1/P2 轉成既有 milestone 的可追蹤項。
  - 驗收：每個新 gate 都能連回 `improve-test.md` 指令或人工驗收。

- [x] **建立最新 acceptance report**
  - 以真實模型 smoke + ASR + metrics 為證據，不只寫「可產生 WAV」。
  - 驗收：每個通過項包含日期、commit、命令、輸出路徑、ASR 結果與殘留風險。
  - 狀態：已新增 `harness/write_acceptance_baseline.ps1`，可包住 quality gate 並輸出
    `baseline_manifest.json` / `baseline_report.md`，也支援 `-RunAsr` 與 `-UpdateAcceptanceReport`；
    已用 CUDA + ASR 跑真實模型 baseline，seed 102 的 Mandarin/clone required-term 檢查全部通過。
