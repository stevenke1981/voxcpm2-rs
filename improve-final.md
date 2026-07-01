# VoxCPM2 Rust/Candle 改善最終驗收

## 參考基準

- Current repo：`E:\voxcpm2_rust_candle_pack`，已用 CBM 建立 `cbm+voxcpm2_rust_candle_pack` 索引。
- Reference repo：`stevenke1981/voxcpm-cpp` shallow clone commit `896f595`，已用 CBM 建立 `cbm+voxcpm-cpp-ref` 索引。

## 本次診斷結論

Rust/Candle 版本已經具備真實語音生成與 voice clone 的核心能力；CBM 顯示目前主要路徑集中在：

- `crates/voxcpm2-core/src/pipeline.rs`：`VoxPipeline`、`SynthRequest`、model cache、synth/clone pipeline、reference prefix encoding。
- `crates/voxcpm2-core/src/audio.rs`：generated speech polish、clone reference polish、adaptive background gate、harsh midband smoother。
- `crates/voxcpm2-cli/src/main.rs`：synth/clone/inspect/benchmark CLI。
- `crates/voxcpm2-gui/src/main.rs`：GUI worker/cancel/update loop。

與 `voxcpm-cpp` 相比，Rust 目前的主要缺口不是「能不能出聲」，而是：

1. voice clone consent 尚未形成 CLI/GUI/文件/測試的完整 release gate。
2. clone 模式尚未完整覆蓋 reference-only、prompt-only continuation、combined。
3. 音訊品質修正尚未形成固定 WAV metrics + ASR + backend matrix 的 release gate。
4. GUI 與 CLI 還需要共用同一套 safety/quality gate。
5. release package hygiene 尚未像 C++ repo 一樣明確排除模型、WAV、fixtures、debug dump。

## 2026-07-01 進度更新

已完成的下一步改善：

- CLI clone 已強制 `--i-have-consent`，未授權時不載入模型也不產生 WAV。
- `synth` / `clone` 已支援 `--metrics-out`，可保存 speech polish 與 clone reference polish JSON。
- `harness/audio_quality_gate.ps1` 已可跑 seed 99 Mandarin prompt set、clone dry-run、WAV 結構/headroom gate，
  並輸出 frame-level `*.metrics.quality.json`。
- GUI clone tab 已加入 reference voice consent checkbox；GUI synth/clone 會建立 metrics sidecar。
- 繁體中文語言風險提示已移到共用 `synthesize()` 入口，dry-run CLI smoke 也會提示；
  簡體中文 Mandarin 基準不誤觸。
- Clone reference 已有 clean/noisy/long synthetic fixture gate，並在 clone metrics 中輸出
  `clone_reference_trim` 以記錄長 reference 裁切診斷。
- Accepted baseline writer 已新增：`harness/write_acceptance_baseline.ps1` 可產生
  `baseline_manifest.json` 與 `baseline_report.md`，並可透過 `-RunAsr` 將 ASR transcript 與
  required-term 檢查納入 release evidence。
- 真實模型 CUDA + ASR accepted baseline 已完成：seed 102 通過 Mandarin short、midburst、
  fricatives 與 clone 四個 required-term gate；seed 99 仍保留為 regression probe，但本次短句
  ASR 未通過，因此不作為 accepted baseline。
- Quality sweep harness 已新增：`harness/quality_sweep.ps1` 可用 ASR similarity、required terms、
  high-ZCR、quiet RMS、clone reference floor 與 peak penalty 比較 seed/cfg/latent_norm/scheduler。
- Targeted CUDA + ASR sweep 已跑 seed 102、cfg 2.3/2.5、latent default/0.7875、uniform scheduler，
  Mandarin short 與 midburst 兩句皆 ASR similarity=1.0 且 required terms 通過；目前最佳候選是
  `seed102_cfg2p5_lndefault_uniform`，但尚未含 clone/full prompt matrix，因此先不取代 accepted baseline。
- 已對齊官方 Python `OpenBMB/VoxCPM@b9fbaec` 與 C++ `voxcpm-cpp@896f595` 的 clone/語音生成
  序列基礎：CLI 支援 reference-only、prompt-only continuation、reference+prompt combined；
  pipeline 依官方順序組出 ref prefix、`prompt_text + target_text + audio_start`、prompt patches，
  並用 prompt 最後一個 latent patch 初始化 CFM condition。
- 追蹤官方最新 `OpenBMB/VoxCPM@07c937b` 後，補齊兩個細節：audio patch placeholder token
  改為官方/C++ 使用的 `0`；autoregressive `max_len` 改用 target text token 長度，而不是
  combined sequence 長度，避免 reference/prompt patches 放大生成時間。
- 已產生 CUDA 測試語音：
  `output/alignment_synth_short_seed102.wav`、`output/alignment_synth_seed102.wav`、
  `output/alignment_prompt_only_seed102.wav`、`output/alignment_combined_seed102_v2.wav`。
  短句 synth 經 faster-whisper large-v3-turbo CUDA ASR 通過主要文字：
  「这是普通话测试 / 声音清楚自然」。

## 最終完成定義

以下條件全部通過後，才可宣稱 Rust/Candle 版本達到「可正常產生清楚人聲語音、可安全複製語音、可發布」：

- **Safety**：clone 必須要求明確 consent；未授權 clone 不生成 WAV。
- **Speech quality**：seed 99 Mandarin synth 通過 WAV metrics 與 ASR；中段 high-ZCR / 4-8kHz burst 不回歸。
- **Clone quality**：reference noise 在 AudioVAE encoder 前被量測並降低；clone WAV 通過 ASR 與 metrics。
- **Parity**：clone sequence、padding、mask、patch count 與 Python/C++ fixture 對齊。
- **Backend**：CPU/CUDA smoke 都產生 finite、non-empty、無 clipping 的 WAV，且紀錄 RTF/記憶體。
- **GUI**：GUI synth/clone/cancel 與 CLI 使用同一套安全、模型與音訊診斷。
- **Release**：release build/package 不包含模型權重、WAV、fixtures、debug artifacts，文件與實測一致。

## 建議交付順序

1. 先修 P0 consent gate，因為這是安全與產品契約，不依賴模型權重。
2. 接著把 `AudioPolishReport` 串成 JSON/metrics artifact，讓後續音質修正都可比較。
3. 建立 Mandarin seed 99 synth gate 與 noisy reference clone gate。
4. 補 clone parity fixtures，再擴展 prompt-only/combined clone。
5. 最後做 backend matrix、GUI smoke、release package hygiene。

## 風險

- 不應再用更重的全域低通或 gate 嘗試消除所有雜音；目前 lessons 指向中段刺耳音通常是 high-ZCR / 4-8kHz burst，需要局部 smoother 與 ASR 保護。
- Mandarin 品質測試若使用繁體中文，可能被 VoxCPM2 判為廣東話傾向；基準測試必須使用簡體中文。
- CFM RNG 與 Python/CUDA 不同，不能用 same seed 期待 sample-level 完全一致；需要固定 noise tensor fixture 才能做嚴格 parity。
- Voice clone reference 若太長會放大 AudioVAE encoder 記憶體壓力；trim 行為要明確記錄並測試。

## 下一個 commit 建議

下一個實作 commit 建議只做一件事：改善 prompt-only/combined clone 的 ASR 細節失真與 stop
head 穩定性。prompt-only 目前可辨識主要語意但字詞仍偏移；combined 已被 target-length cap
限制到 30.4 秒，但 faster-whisper 對該輸出仍會 CUDA/Python crash，需要先切出短窗或替換
GPU ASR engine 再做完整判讀。
