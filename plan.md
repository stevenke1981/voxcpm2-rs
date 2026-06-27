# plan.md — 分階段實作計畫

## Phase 0 — 基線與資料固定

目標：建立可追蹤的 source-of-truth。

工作：

1. 下載官方 repo 與 HF 模型。
2. 保存官方 Python generate 的輸出樣本。
3. 建立 manifest：config、tokenizer、safetensors、audiovae。
4. 建立 Python golden runner，輸出每層 shape / dtype / sample audio。

交付：

- `models/VoxCPM2/model_manifest.json`
- `tests/golden/python_outputs/*.json`
- `tests/golden/python_outputs/*.wav`

Gate：官方 Python 能產生穩定 wav。

## Phase 1 — Rust workspace + harness

目標：確保專案可編譯、可測、可跑 smoke。

工作：

1. 建立 `voxcpm2-core`、`voxcpm2-cli`、`voxcpm2-gui`。
2. 實作 config parser、device selector、wav writer。
3. CLI `synth --dry-run` 產生 sine/silence smoke wav。
4. egui 可開啟、可填模型路徑、可呼叫 smoke synth。

Gate：`cargo test --workspace --features cpu` 成功。

## Phase 2 — 模型轉換與載入

目標：Candle 可讀模型資產。

工作：

1. 使用 Candle 讀取 `model.safetensors`。
2. 轉換 `audiovae.pth` → `audiovae.safetensors`。
3. 建立 tensor name mapping 表。
4. 比對 dtype、shape、缺失張量。

Gate：manifest 與 mapping 100% 對齊，無未知必要張量。

## Phase 3 — Tokenizer parity

目標：Rust tokenization 與官方 Python 對齊。

工作：

1. 使用 `tokenizers` crate 讀 `tokenizer.json`。
2. 補上 VoxCPM2 custom tokenizer 的中文 multi-char 行為。
3. 建立 20～100 條固定文字測試。
4. 比對 BOS/EOS、special tokens、chat template。

Gate：token IDs 全部與 Python golden 一致。

## Phase 4 — TSLM / RALM port

目標：完成 MiniCPM-like LM backbone 與 residual acoustic LM。

工作：

1. 實作 RMSNorm、RoPE/LongRoPE、QKV projection、GQA attention、MLP。
2. 實作 KV cache。
3. 實作 muP scale：`scale_emb`、`scale_depth`、`dim_model_base`。
4. shape test 與 Python hidden-state parity。

Gate：固定輸入 hidden state 誤差低於門檻。

## Phase 5 — LocEnc / LocDiT / Flow Matching

目標：完成 diffusion acoustic generation。

工作：

1. 實作 LocEnc。
2. 實作 DiT block、cfg guidance、Euler solver。
3. 實作 `inference_timesteps`、`cfg_value`、seed。
4. 對齊 Python latents。

Gate：latent shape、範圍、統計量與 Python 對齊。

## Phase 6 — AudioVAE V2 decoder

目標：從 latent decode 到 48kHz wav。

工作：

1. 實作 encoder/decoder block。
2. 實作 upsample rates 與 super-resolution path。
3. 產生 wav，檢查 NaN/Inf、peak、duration。
4. 對比官方輸出音訊長度與頻譜。

Gate：可生成可播放語音，不再是雜音。

## Phase 7 — GPU optimization

目標：達到實用速度與記憶體控制。

工作：

1. CUDA/Metal build matrix。
2. BF16/FP16 fallback。
3. attention KV cache memory profiling。
4. streaming chunk decode。
5. 加入 benchmark CLI 與 GUI diagnostics。

Gate：8GB VRAM 等級 GPU 可跑短句；RTX 4090 做 RTF 基準。

## Phase 8 — egui 完整產品化

目標：可給一般使用者操作。

工作：

1. 模型下載/檢查頁。
2. 文字轉語音頁。
3. Voice Design / Cloning 頁。
4. 播放器、波形、匯出。
5. 背景 worker thread + channel + cancel。
6. 錯誤診斷與安全提醒。

Gate：GUI 可完整完成一次真實生成。

## Phase 9 — Release

工作：

1. Windows zip release。
2. Linux AppImage/portable tar。
3. macOS app bundle。
4. README、FAQ、troubleshooting。
5. smoke/golden/benchmark 報告。

Gate：`docs/final.md` 所列交付檢查全數通過。
