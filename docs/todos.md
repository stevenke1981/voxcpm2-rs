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

## Milestone I：egui

- [ ] Model tab。
- [ ] Synthesis tab。
- [ ] Voice Design tab。
- [ ] Cloning tab。
- [ ] Output player。
- [ ] Diagnostics tab。
- [ ] Background worker thread。
- [ ] Cancel generation。

## Milestone J：Release

- [ ] Windows portable zip。
- [ ] README 快速開始。
- [ ] FAQ：CUDA、C4819、缺檔、VRAM。
- [ ] Golden report。
- [x] CLI voice clone consent gate：`clone` 必須帶 `--i-have-consent`，未授權請求會在載入模型或寫 WAV 前失敗。
- [ ] Safety notice。
