# todos.md — OpenCode/Codex 任務清單

## Milestone A：可編譯骨架

- [x] 建立 Rust workspace。
- [x] `voxcpm2-core`：config parser。
- [x] `voxcpm2-core`：device selection。
- [x] `voxcpm2-core`：wav writer。
- [x] `voxcpm2-cli`：`synth --dry-run`。
- [x] `voxcpm2-gui`：egui 基本視窗。
- [x] `harness/run_all.ps1` 可跑完。
  - 已驗證：`cargo fmt`、`cargo clippy`、`cargo test`、`synth --dry-run` 全部通過。

## Milestone B：模型資產工具

- [x] 下載模型到 `models/VoxCPM2`（已使用 `scripts/download_model.py` 完成）。
- [x] Rust `inspect` 指令讀 manifest + safetensors 解析。
  - 支援 tensor 名稱、shape、dtype 分布、總參數量（577 tensors, BF16, 2.29B params）。
- [x] 缺少檔案時輸出修復指令。
- [x] 轉換 `audiovae.pth` → `audiovae.safetensors`（312 tensors, F32, 94.2M params）。
- [x] 生成 `model_manifest.json` + `audiovae_manifest.json`。

## Milestone C：Tokenizer (已全部完成)

- [x] 讀 `tokenizer.json`（經由 `tokenizers` crate）。
- [x] 讀 `tokenizer_config.json` special tokens → `SpecialTokens` struct。
- [x] 實作 chat template（專用 `render_chat_template()` 函數）。
- [x] 實作 VoxCPM2 custom tokenizer CJK multi-char splitting（`build_cjk_split_map()` + `expand_cjk()`）。
- [x] 與 Python token IDs 對齊（`tokenizer_parity_python` 測試，14 個案例全部通過）。

關鍵發現：VoxCPM2 使用 LlamaTokenizerFast + CJK 拆分（multi-char Chinese token → 單字 split），BOS=`<s>`(id=1)，EOS=`<|im_end|>`(id=73440)，vocab_size=73440+added_tokens。

## Milestone D：TSLM（模組實作完成，測試通過）

- [x] 實作 RMSNorm。
- [x] 實作 LongRoPE（含 RopeScaling config）。
- [x] 實作 GQA attention（含 KVCache，Option<Tensor> + cat 模式）。
- [x] 實作 MLP / activation。
- [x] 實作 KV cache。
- [x] Layer 0 golden test：所有 10 個中間張量 cosine similarity = 1.0（通過）。
  - after_input_layernorm、q/k/v_proj_out、q/k_rope_out、attn_output 等。
- [x] 發現 crate RMSNorm bug：`mean_keepdim(1)` 在 3D [B,T,D] 輸入時，會對序列維度 T 做歸一化，而非特徵維度 D（應使用 `mean_keepdim(-1)`）。
- [ ] 修正 crate 的 RMSNorm forward（改為支援任意維度）。

## Milestone E：RALM / LocEnc（模組實作完成，測試通過）

- [x] 實作 residual LM 8 層（無 RoPE，manual GQA forward）。
- [x] 實作 acoustic feature patching：`patch_size = 4`。
- [x] 實作 `feat_dim = 64`。
- [ ] 實作 scalar quantization scale/latent dim（需 Python insight）。

## Milestone F：LocDiT + Scheduler（模組實作完成，測試通過）

- [x] 實作 DiT block。
- [x] 實作 cfg guidance。
- [x] 實作 Euler solver + FlowMatchingScheduler。
- [x] 支援 `inference_timesteps`（含 uniform/log-norm scheduler）。
- [x] 支援 deterministic seed（CPU fallback graceful）。
- [ ] 對齊 real model shapes（hidden_dim=1024, feat_dim=64, 12 layers）。

## Milestone G：AudioVAE（結構完成，待 weight load）

- [x] 實作 AudioVAE decoder 結構（Conv1d with weight norm）。
- [x] 實作 AudioVAE encoder 結構。
- [ ] 載入 `audiovae.safetensors` 真實權重。
- [ ] 輸出 48kHz wav（解碼 pipeline）。
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
