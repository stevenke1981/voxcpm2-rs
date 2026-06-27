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
- [ ] Safety notice。
