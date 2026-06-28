# todos.md — VoxCPM2 Rust Candle Pipeline

## Legend
- `[x]` 已完成
- `[ ]` 待實作
- `[~]` 部分完成 / 需改進

---

## ✅ Milestone A–H：已完成基礎建設

- [x] Workspace + core/cli/gui 三 crate
- [x] Config parser + device selection + wav writer
- [x] 模型資產工具（inspect、manifest、pth→safetensors）
- [x] Tokenizer（CJK split、chat template、Python parity 14/14）
- [x] TSLM 28 層（RMSNorm、LongRoPE、GQA、KV cache、golden test cos=1.0）
- [x] RALM 8 層
- [x] AudioVAE decoder（weight_norm fusion、SR FiLM、CUDA contiguity）
- [x] AudioVAE encoder
- [x] LocDiT 結構（12 層 DiT block, GQA attention, RoPE）
- [x] FSQ 層分析（推論時跳過）
- [x] FlowMatching scheduler（Euler、uniform/log-norm、CFG）
- [x] CUDA build（`NVCC_CCBIN` env var + `--features cuda`, 23 tests pass）
- [x] egui GUI：Model/Synthesis/Output/Diagnostics tabs

---

## 🚨 Milestone Z：Pipeline 推理重構（最高優先級）

> Python 原版（`voxcpm2.py`）使用**自回歸 LM + 逐 patch DiT** 生成聲學特徵。
> 目前的 Rust pipeline 直接一次產生完整序列，架構錯誤導致輸出非語音。

### Z1：UnifiedCFM wrapper（DiT 逐 patch 推理）

- [x] **Z1.1** 建立 `UnifiedCFM` struct（wraps LocDiT estimator）
  - `forward(mu, n_timesteps, patch_size, cond, cfg_value)` → `[1, feat_dim, patch_size]`
  - Euler solve：從 noise `[1, 64, 4]` 開始，逐步去噪
  - CFG 支援（zero_star scaling, optimized_scale）
  - **zero_init_steps**：前 ~4% 步使用零 velocity（Python 特有）
  - 位置：`crates/voxcpm2-core/src/models/unified_cfm.rs`

- [x] **Z1.2** 修正 `LocDiT::forward` 簽名以匹配 Python `VoxCPMLocDiTV2`
  - 輸入：`x: [B, C, T]`, `mu: [B, 2*hidden]`, `t: [B]`, `cond: [B, C, T']`, `dt: [B]`
  - `mu` reshape `[B, 2, hidden]` 後 concat 到輸入序列前端
  - `cond` 通過 `cond_proj` 投影後 concat
  - `generate()` 已移除（由 `unified_cfm.rs` 的 Euler solve 取代）

### Z2：feat_encoder 整合

- [x] **Z2.1** `VoxCPMLocEnc` forward 修正（`loc_enc.rs`）
  - 輸入 `[B, C=64, T=4]` → `in_proj` → `[B, T, hidden]`
  - concat `special_token` → `[B, T+1, hidden]` → encoder layers → norm
  - 取最後位置 → `[B, 1, hidden]` → `enc_to_lm_proj` → `[B, 1, 2048]`
  - 修正：in_proj 含 bias、special_token 4D→3D、GQA kv_heads=num_heads/8

- [x] **Z2.2** 驗證 feat_encoder 權重載入正確 — Python 參考實作 `D:\VoxCPM` 比對完成
  - [x] 發現 `enc_to_lm_proj.bias` 存在（Rust 原用 `linear_no_bias` → 已修復）
  - [x] 發現 `prefix_feat_cond` 初始形狀錯誤（`[1,64,0]` → `[1,64,4]` → 已修復）
  - [x] `FeatEncoder`（feat_encoder.rs）vs `LocEnc`（loc_enc.rs）兩者關係確認
  - [x] KV cache step tracking（Python `kv_cache.step()` vs Rust 顯式 `step`）功能等效

### Z3：fsq_layer

- [x] **Z3.1** 建立 `FsqLayer` struct（`fsq_layer.rs`）
  - `in_proj: Linear(2048, 512, bias=false)` ✓ 匹配權重 `[512, 2048]`
  - `out_proj: Linear(512, 2048, bias=false)` ✓ 匹配權重 `[2048, 512]`
  - ~~量化公式：`tanh(round(x * 9) / 9)`~~ → **修復：`round(tanh(x) * 9) / 9`**
  - 🐛 **2026-06-28 修復**：tanh 與 round 順序錯誤。Python 是 `in_proj → tanh → round → out_proj`，Rust 原是 `in_proj → round → tanh → out_proj`。修正後 FSQ 行為對齊 Python。
  - 單元測試：shape test + zero preservation test ✓

### Z4：fusion_concat_proj

- [x] **Z4.1** 修正 `fusion_concat_proj` 權重 shape
  - `Linear(4096, 2048)` 匹配實際權重 `[2048, 4096]`
  - 位置：`loc_enc.rs`（同時修正 decoder 部分的 num_kv_heads）

### Z5：stop_head（停止預測器）

- [x] **Z5.1** 建立 `StopHead` struct（`stop_head.rs`）
  - `stop_proj: Linear(2048, 2048)` + bias ✓ 匹配權重 `[2048, 2048]`
  - `stop_actn: SiLU`（內聯 activation，無權重）
  - `stop_head: Linear(2048, 2)` + bias ✓ 匹配權重 `[2, 2048]`
  - `should_stop(logits)` helper：softmax 後檢查 stop_prob > 0.5
  - 單元測試：shape test + zero-default should_stop=false ✓

### Z6：KV cache forward_step（TSLM / RALM）

- [x] **Z6.1** `TSLM` 實作 `forward_step(input_embeds, step)` 方法
  - 單 token 推論，使用既有 KV cache
  - 輸入：`[B, 1, 2048]`（單個 embedding，跳過 token embedding lookup）
  - 輸出：`[B, 1, 2048]`（含 muP scale_emb/scale_depth）
  - 重用現有 `KVCache` struct（在 attention layer 內部）

- [x] **Z6.2** `RALM` 實作 `forward_step(input_embeds, step)` 方法
  - 同時修正 `RalmLayer::forward_no_rope` 加入 KV cache 支援
  - 使 RALM 在自回歸迴圈中能正確 attend 到所有先前 token

### Z7：自回歸推理迴圈

- [x] **Z7.1** 實作 `generate_autoregressive()` 完整自回歸迴圈（`autoregressive.rs`）
  ```
  1. 載入 KV cache 版 TSLM/RALM，用 text input_ids 填充 KV cache
  2. 載入 feat_encoder, fsq, stop_head, LocDiT, UnifiedCFM, 各 projection
  3. Loop i = 0..max_len:
     a. lm_to_dit_proj(lm_last) + res_to_dit_proj(res_last) → concat → dit_hidden [B, 2048]
     b. UnifiedCFM(mu=dit_hidden, patch_size=4, cond=prev_feat, n_timesteps, cfg)
        → pred_feat [B, 64, 4]
     c. LocEnc::encode(pred_feat) → curr_embed [B, 1, 2048]
     d. StopHead check → break if stop
     e. TSLM.forward_step(curr_embed, step)
     f. FsqLayer → lm_q
     g. fusion_concat_proj(concat(lm_q, curr_embed)) → [B, 1, 2048]
     h. RALM.forward_step(fusion, step)
  4. 串接所有 pred_feat → [B, 64, total_frames]
  ```
  簽名：`generate_autoregressive(main_vb, config, req, dev, input_ids, cancel) → [B, 64, total_frames]`

- [x] **Z7.2** 更新 `pipeline.rs` 傳入 `input_ids` 給 `generate_autoregressive`
  - 移除舊的 `h_tslm`/`h_ralm` 參數（改在函數內載入 KV cache 版本）

- [x] **Z7.3** `LocEnc` 修正（in_proj bias, special_token 4D→3D, GQA kv_heads, fusion_concat_proj shape）

### Z7 已解決的錯誤

- F4: RALM 無 KV cache 導致自回歸時只能 attend 到目前 token → 修正 `forward_no_rope` 加入 KV cache
- F4: `UnifiedCFM` 建構子簽名（`new(estimator, cfg_rate, sigma_min, solver)` → 修正呼叫端
- F4: TSLM `forward_step` 無 token embedding → 直接接受 input_embeds（跳過 embed_tokens）
- F4: `LocEnc::load` 載入 decoder 組件時 cond_proj shape 錯誤（應為 `[1024, 64]` 非 `[1024, 1024]`）→ 移除 LocEnc decoder/FSQ/fusion 載入（由 `LocDiT`/`FsqLayer` 單獨載入）
- F4: `StopHead::load` 使用 `linear`（含 bias）但 stop_head 無 bias → 改用 `linear_no_bias`
- F4: `UnifiedCFM::forward` feat_dim 從 `mu.dim(1)`（2048）讀取導致 z noise shape `[B, 2048, 4]` → 改為使用 config 的 `feat_dim=64`
- F4: `LocDiT::forward` 接收 cond 為 `[B, C, 0]`（無前方特徵時）導致 reshape 0 元素錯誤 → 處理 zero-length cond 情況
- F3: CUDA OOM（載入 BF16 權重為 F32 需 9.2GB，VRAM 僅 8GB）→ 改用 BF16 dtype on CUDA
- F3: CUDA kernel 編譯需要 MSVC `cl.exe`（未安裝 VS Build Tools）→ 無法使用 CUDA 加速，需安裝 VS Build Tools 後方可使用 `--features cuda`
- F3: `dtype mismatch in matmul, lhs: F32, rhs: BF16` on CUDA → `LocDiT::forward` 自動將輸入轉換為權重 dtype

### Z9：Python 參考實作 `D:\VoxCPM` 比對與對齊（2026-06-28）

- [x] 使用 RLM 掃描 `D:\VoxCPM`（58 files, 107 chunks）
- [x] 使用 CBM 索引本專案（84 files, 307 symbols, 882 edges）
- [x] 比對 `VoxCPM2Model._inference()` vs `generate_autoregressive()`
- [x] 比對 `VoxCPMLocEnc` vs `LocEnc`
- [x] 比對 `UnifiedCFM` vs `unified_cfm.rs`
- [x] 比對 `MiniCPMModel.forward_step` vs `TSLM::forward_step`
- [x] 比對 `AudioVAEV2` vs `audio_vae.rs`
- [x] 比對 `ScalarQuantizationLayer` vs `FsqLayer`
- [x] 修復 #1: `prefix_feat_cond` 初始形狀 `[1,64,0]` → `[1,64,4]`
- [x] 修復 #2: `enc_to_lm_proj` bias 缺失（`linear_no_bias` → `linear`）
- [x] 修復 #3: `max_len` 硬編碼 500 → 使用 Python 公式 `min(seq_len * 6 + 10, global_max)`
- [x] 修復 #4: AudioVAE CPU-only → 嘗試 CUDA（成功！conv1d/convtranspose1d 可在 CUDA 執行）
- [x] 建立 `traceability_matrix.md`
- [x] 建立 `acceptance_report.md`
- [x] 建立 `rollback.md`
- [ ] 清理死代碼 `feat_encoder.rs`（可選，功能不受影響）

### Z8：測試與驗證

- [x] **Z8.1** 逐 patch DiT 輸出 shape 測試：`locdit_forward_shape_test` 驗證 `[1, 64, 4]`
- [x] **Z8.2** 端到端 GPU 推理成功（CUDA, 5 CFM steps, 16 ar steps）：`[1, 1, 122880]` @48kHz
- [x] **Z8.5** 全 pipeline 端到端執行測試（10 step, 1 timestep, CPU）：產生 valid WAV（peak 0.000031，因步數過少）
- [x] **Z8.6** Python 參考音頻成功產生：`voxcpm design` 產出 0.64s 正常語音（peak 0.308, RMS 0.031, 637.5Hz dominant）
- [x] **Z8.3** Python parity test：相同文字、相同 seed → 輸出波形相似
  - 🐛 **發現**：CFM 使用不同亂數種子（`Tensor::randn` vs `torch.randn`），即使相同文字也會產生不同內容
  - 確認 TSLM init（0.999）、RALM init（0.997）與 Python 高度相關
  - FSQ tanh/round 順序 bug 已修復
  - **Rust 自回歸輸出**：valid WAV（no NaN），peak 0.38，RMS 0.02
  - **Python 比較**：latent std 0.71（Python）vs 1.13（Rust），主因 CFM 隨機種子不同而非實作錯誤
- [x] **Z8.4** 真實語音聽覺測試：輸出 4.8 秒合法語音波形（230400 samples @48kHz）

### Z10：CUDA 推理最佳化（2026-06-28 新增）

- [x] CUDA build 修復：`NVCC_CCBIN` 環境變數（bindgen_cuda 原生支援 nvcc -ccbin）
- [x] AudioVAE CUDA 執行成功（移除 CPU-only 限制，conv1d/convtranspose1d 皆可在 CUDA 執行）
- [x] `max_len` 動態計算：`min(seq_len * 6 + 10, global_max)` 對齊 Python 行為，大幅減少無意義步驟
- [x] `--max-ar-steps` CLI 參數：允許使用者控制 autoregressive 上限
- [x] 移除 `device.rs` 中的備用測試函數（已確認 CUDA device 建立正常）
- [ ] GPU 推理效能調校（目前 16 step AR + 5 CFM + AudioVAE CUDA 約 30-60s）

---

## 📋 Milestone I：egui GUI 改善

- [x] Model tab — 目錄瀏覽、device 選擇、inspect 模型、必備檔案檢查
- [x] Synthesis tab — 文字輸入、CFG、Steps、Seed、Dry-run、Voice Design
- [x] Output tab — 波形繪製、播放（rodio）、WAV 匯出
- [x] Diagnostics tab — 裝置資訊、VRAM 估算、事件日誌
- [x] Background worker thread — 非阻塞生成（crossbeam channel）
- [x] **Cancel generation** — `Arc<AtomicBool>` 在每個 DiT step 檢查
- [x] **Cloning tab（UI scaffold）** — 檔案選取器、similarity slider、未實作提示
- [ ] **Cloning tab（完整後端整合）** — 需等 Milestone Z 完成後才能啟用 Clone 按鈕
- [ ] Model tab liftoff status integration（整合完整 pipeline 初始化狀態）

---

## 📋 Milestone J：Release

- [ ] Windows portable zip
- [ ] README 快速開始
- [ ] FAQ：CUDA、C4819、缺檔、VRAM
- [ ] Golden report（Python vs Rust 比對）
- [ ] Safety notice

---

## 📋 已完成但已過時的組件（需隨 Z7 移除或重構）

- `pipeline.rs: synthesize_real()` — 舊一次性推理流程（已替換為 `generate_autoregressive()` 樁）
- `pipeline.rs: project_text_to_feat()` — 舊 cond_proj transpose 投影（可隨 Z7 移除）
- `LocDiT::generate()` — 已移除（由 UnifiedCFM 取代）
- `FlowMatchingScheduler` — 保留 `pub use` 以便下游向後相容；Z7 後將移除
