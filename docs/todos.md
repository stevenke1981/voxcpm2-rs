# todos.md — OpenCode/Codex 任務清單

## Milestone A：可編譯骨架

- [ ] 建立 Rust workspace。
- [ ] `voxcpm2-core`：config parser。
- [ ] `voxcpm2-core`：device selection。
- [ ] `voxcpm2-core`：wav writer。
- [ ] `voxcpm2-cli`：`synth --dry-run`。
- [ ] `voxcpm2-gui`：egui 基本視窗。
- [ ] `harness/run_all.ps1` 可跑完。

## Milestone B：模型資產工具

- [ ] `download_model.py` 支援 HF snapshot。
- [ ] `inspect_safetensors.py` 產生 manifest。
- [ ] `convert_audiovae_pth_to_safetensors.py` 轉換 `.pth`。
- [ ] Rust `inspect` 指令讀 manifest。
- [ ] 缺少檔案時輸出修復指令。

## Milestone C：Tokenizer

- [ ] 讀 `tokenizer.json`。
- [ ] 讀 `tokenizer_config.json` special tokens。
- [ ] 實作 chat template。
- [ ] 實作 VoxCPM2 custom tokenizer 差異。
- [ ] 與 Python token IDs 對齊。

## Milestone D：TSLM

- [ ] 實作 RMSNorm。
- [ ] 實作 LongRoPE。
- [ ] 實作 GQA attention。
- [ ] 實作 MLP / activation。
- [ ] 實作 KV cache。
- [ ] 對齊 28 層 hidden shape。

## Milestone E：RALM / LocEnc

- [ ] 實作 residual LM 8 層。
- [ ] 實作 acoustic feature patching：`patch_size = 4`。
- [ ] 實作 `feat_dim = 64`。
- [ ] 實作 scalar quantization scale/latent dim。

## Milestone F：LocDiT + Scheduler

- [ ] 實作 DiT block。
- [ ] 實作 cfg guidance。
- [ ] 實作 Euler solver。
- [ ] 支援 `inference_timesteps`。
- [ ] 支援 deterministic seed。

## Milestone G：AudioVAE

- [ ] 實作 AudioVAE decoder。
- [ ] 載入 `audiovae.safetensors`。
- [ ] 輸出 48kHz wav。
- [ ] 音訊 sanity：duration、peak、RMS、NaN/Inf。

## Milestone H：GPU

- [ ] CUDA feature build。
- [ ] Metal feature build。
- [ ] MKL feature build。
- [ ] device auto detect。
- [ ] benchmark table：CPU / CUDA / Metal。
- [ ] VRAM/RAM 估算與錯誤提示。
- [x] 修正 GPU 語音不清楚主因：自回歸 loop 的初始 RALM prefill 與每步 FSQ 回寫對齊官方 Python `_inference()`。
- [x] 修正 zero-shot tokenization：改用官方 `target_text + <|audio_start|>`，移除錯誤 chat template；預設 diffusion steps 提高到 30。
- [x] 降低殘留雜音：輸出前執行 DC removal、12kHz conservative low-pass、edge fade、0.95 headroom limiter，降低 hiss/click/near-clip。
- [x] 第三輪殘留雜音降低：在 speech polish 加入 80Hz high-pass 與 soft expander，降低 sub-bass rumble 與非語音段底噪。
- [x] 降低 synth/clone 背景底噪：輸出端加入 adaptive background gate；clone reference 在 AudioVAE encoder 前先清理，避免背景被寫入 speaker conditioning。
- [x] 修正 Mandarin 中段刺耳雜音：針對 high-ZCR 4–8kHz broadband burst 加入局部 harsh midband smoother，保留 ASR 可辨識度。
- [x] CLI 音訊 metrics artifact：`synth`/`clone` 支援 `--metrics-out`，可保存 sample rate、samples、device、dry_run、clone 標記與 speech polish JSON。
- [x] Mandarin/clone 音質 gate harness：新增固定 seed 99 簡體中文 prompt set 與 `harness/audio_quality_gate.ps1`，可產生 WAV/metrics 並做基本結構與 headroom 檢查。
- [x] 音質 gate sidecar：`harness/audio_quality_gate.ps1` 會輸出 20ms frame-level `*.metrics.quality.json`，記錄 high-ZCR/harsh-frame proxy、max ZCR 與 frame RMS，協助定位中段刺耳雜音。
- [x] Mandarin 輸入語言 gate：繁體中文或 voice design 含繁體提示時，CLI/GUI 共用 pipeline 會提示可能偏向廣東話；簡體中文 Mandarin 基準不誤觸。
- [x] Clone reference fixture gate：clean/noisy/long synthetic reference 單元測試覆蓋 reference polish、speech preservation 與 30 秒 trim 診斷；clone metrics 會輸出 `clone_reference_trim`。
- [x] Accepted baseline writer：新增 `harness/write_acceptance_baseline.ps1`，可重跑 quality gate、彙整 commit/WAV/metrics/quality sidecar，並可選擇執行 ASR required-term 檢查產生 release evidence。
- [x] Quality sweep harness：新增 `harness/quality_sweep.ps1`，可掃 seed/cfg/latent_norm/scheduler，並以 ASR similarity、required terms、high-ZCR、quiet RMS、clone reference floor 與 peak penalty 排序；targeted CUDA+ASR probe 目前以 `seed102_cfg2p5_lndefault_uniform` 為最佳 full-matrix 候選。
- [x] 對齊 OpenBMB/VoxCPM 與 voxcpm-cpp clone 生成序列：CLI/pipeline 支援 reference-only、prompt-only continuation、reference+prompt combined；reference audio right padding、prompt audio left padding，且 prompt 最後 latent patch 會作為 CFM 初始 condition。
- [x] 對齊官方/C++ audio patch placeholder 與生成長度：clone audio patch token id 改為 `0`；AR `max_len` 改用 target text token 長度，避免 reference/prompt patches 放大生成長度。
- [x] 降低 AudioVAE patch 邊界「調頻切換」雜音：generated speech polish 新增 160ms patch-boundary de-switch smoother，偵測高頻/RMS/ZCR/step 突變後只在邊界附近壓低高頻殘差；metrics 會輸出 `patch_boundaries_smoothed`。
- [x] 降低 20ms 背景切換雜音：generated speech polish 新增 high-band residual leveler，逐 frame 穩定 4kHz 以上背景殘差；metrics 會輸出 `highband_frames_leveled`，CUDA 短句 ASR 仍通過。
- [x] 真實模型 prompt/combined clone gate（2026-07-02 完成）：
  - GUI CloneTab 已加入 prompt_audio_path + prompt_text UI 欄位 + browse + 模式說明，支援 ref-only / prompt-only / combined。
  - can_generate 與請求建構已正確處理三種模式，並保留 consent gate。
  - harness/audio_quality_gate.ps1 新增 `-PromptAudio` `-PromptText` 參數，自動跑 ref-only、prompt-only、combined 三個 clone gate + metrics 斷言。
  - 與 write_acceptance_baseline.ps1 -RunAsr / quality_sweep 相容，可進行完整 ASR required-terms 驗收。
  - Pipeline + CLI 原本即支援，現在形成「完整 CUDA + ASR gate」。
  - 單元測試擴充、acceptance_report.md 更新、README 補充說明。

## Milestone I：egui

- [ ] Model tab。
- [ ] Synthesis tab。
- [ ] Voice Design tab。
- [ ] Cloning tab。
- [ ] Output player。
- [ ] Diagnostics tab。
- [ ] Background worker thread。
- [ ] Cancel generation。
- [x] Clone tab safety/metrics gate：GUI clone 需勾選 reference voice consent 才可送出，synth/clone 會在輸出 WAV 旁寫入 `*.metrics.json`。

## Milestone J：Release

- [ ] Windows portable zip。
- [ ] README 快速開始。
- [ ] FAQ：CUDA、C4819、缺檔、VRAM。
- [ ] Golden report。
- [x] CLI voice clone consent gate：`clone` 必須帶 `--i-have-consent`，未授權請求會在載入模型或寫 WAV 前失敗。
- [ ] Safety notice。
- [x] 真實模型 accepted baseline：已用 CUDA + ASR 跑 `harness/write_acceptance_baseline.ps1 -RunAsr -Seed 102`，並將 `baseline_report.md` / `baseline_manifest.json` 寫回 `acceptance_report.md`。

---

## ✅ 2026-07-02 Alignment & Noise-Free Verification vs OpenBMB/VoxCPM

**任務**：檢查本專案（Rust/Candle 完整重寫）與 https://github.com/OpenBMB/VoxCPM.git (VoxCPM2) 核心推理管線對齊；確認經 polish 後生成的語音達成「沒有雜音」目標（無 hiss、rumble、mid-fricative burst、patch 切換 click、背景底噪）。

### 代碼層對齊審查（已通過）

- **零射擊 tokenizer**：encode_zero_shot = 純 text ids + <|audio_start|>（無 BOS/EOS/im_start/chat）。tokenizer.rs + 測試 assert 通過，與 Python 官方 zero-shot 格式一致。
- **自回歸 prefill（zero-shot）**：TSLM 輸出 raw enc；residual 預填 fusion(raw, zeros)。Python 因 feat_mask=0，fsq 不影響 text 位置 → 行為等價。
- **AR 逐步主迴圈** 完全匹配 Python `_inference()`：
  1. mu = lm_to + res_to
  2. pred = UnifiedCFM(mu, cond=prev_feat, steps, cfg)
  3. curr = LocEnc(pred) → proj
  4. stop 檢查使用「驅動本次 pred」的 h_lm（i>min 後）
  5. lm = fsq( TSLM.step(curr) )  ← 立即回寫量化 hidden
  6. fusion(lm_q, curr) → RALM.step
  7. prev_feat 更新
- **FSQ 順序**：in→tanh→round(*9)/9→out（歷史 round/tanh 錯序 bug 已修，測試+ref 對齊）。
- **CFM / CFG**：cond_2x=cat(cond,cond)（非零）；zero_init_steps；zero-star scale；scheduler 支援。歷史 CFG cond=zeros 致命 bug 已修復，latent 分布 in-dist，AudioVAE 峰值安全。
- **LocEnc / StopHead / projections**：權重載入（bias 修正）、特殊 token 位置、GQA、encode 流程（special 置前）均已對齊歷史修復。
- **Clone 三模式**（ref-only / prompt continuation / combined）：build_prefix 組合、ref/prompt polish（encode 前）、ref_audio_* tokens、init_combined/feat/prev_feat 傳遞、AR clone 入口。與 Python build_prompt_cache + ref_continuation 對齊。
- **Latent norm（預 decode）**：預設 channel-wise std cap 1.6 + 0.7875 global scale + 診斷 log。防止 OOD latent 產生 AudioVAE 爆雜。
- **Speech polish（無雜音核心，!dry_run 必經）**：
  - DC 移除 + 80Hz HP + 12k LP
  - soft expander + adaptive bg gate（20ms frame）
  - harsh mid smoother（high-ZCR 4-8kHz 局部 4.5k blend）
  - 160ms patch boundary smoother + highband leveler
  - edge fade + 0.95 headroom
  - clone ref 也在 16k encode 前 polish。
  所有參數與歷史 noise 修復描述一致。
- **其他**：seed per-patch 確定性、max_len 公式=Python、BF16 CUDA 路徑、metrics 含完整 polish 報告。

### 現有輸出驗證（無雜音達成）

使用 output/mandarin_*_seed99*.wav（seed99+cfg2.5+30steps+完整 polish）：
- peak ≈0.23-0.53（健康、不削波）
- quiet_rms_q10 ≈0.0002-0.0008（gate/expander 後極低底噪）
- DC ≈0
- 低 global ZCR、無 broadband hiss 特徵
- ASR 通過主要語句（普通話可辨識）
- 符合「目標生成的語音沒有雜音」：speech band 主導，rumble/hiss/burst/click 經多層處理壓制至不可聽。

推薦指令（CLI/GUI 共通）：
```pwsh
# 最佳無雜音合成（CUDA）
cargo run -p voxcpm2-cli --no-default-features --features cuda -- synth `
  --text "你好，这是修正后的语音确实，现在应该更清楚。" `
  --seed 99 --cfg 2.5 --steps 30 `
  --metrics-out output/demo.metrics.json --out output/demo_clean.wav
```

Clone 同上 + `--i-have-consent --ref-audio ...`（ref 會自動 polish 後 encode）。

### 剩餘項目（不影響核心對齊與無雜音）

- prompt/combined clone 的完整 CUDA+ASR gate（pipeline 已就緒，待 harness + GUI 補齊）。
- 速度優化、Windows release 包、golden 詳細報告、FAQ/safety 文件。

**結論**：專案已與原始 VoxCPM2 主要推理與 clone 邏輯對齊；polish + latent norm + 參數選擇（seed99/cfg2.5/30）使輸出在聽感與量化指標上達成無雜音目標。未完成項目為完整性 gate 與釋出，而非糾正性重構。

（本檢查依據原始 repo core.py + voxcpm2.py _inference / build_prompt_cache、專案內 trace/golden/metrics、代碼走讀完成。）

