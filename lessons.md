# lessons.md — 耐久經驗教訓

## AudioVAE 解碼器對隨機輸入輸出常數（2026-06-27）

**現象**：AudioVAE 解碼器對任何隨機 latent（不同 scale、不同 seed）都輸出相同的常數值
（~-7.25e-05 peak）。PyTorch 中也驗證相同行為。

**原因**：解碼器各層的 alpha gating + 非常小的 weight_g（模型 9 的 weight_g=0.0154）
導致訊號在各個 upsampling block 中逐步衰減。最終的 output 不受輸入影響，這是正確的設計
——解碼器只在輸入與 AudioVAE 編碼器訓練分布匹配時才產生結構化輸出。

**教訓**：
1. 不該用隨機 latent 測試 AudioVAE 輸出品質——它永遠會輸出常數。
2. 模型的正確性測試應專注於 shape、數值精度（Rust vs PyTorch matching），而非輸出聽感。
3. `--gain 100.0` 是必要參數，非可選——因為 DiT 當前的 latent 分布與 encoder 不匹配。
4. Full decoder PyTorch matching test 的正確驗證方式是：給兩個完全相同（seed=42）的輸入，
   比較 peak 值（0.000051 in both），而非檢查輸出是否像語音。

## CUDA contiguity 問題（2026-06-27）

**現象**：CUDA 上 `transpose()` 後的 tensor 做 `matmul()` 會 crash。

**原因**：Candle 的 CUDA backend 要求 matmul 輸入連續。PyTorch 會自動處理 stride，
Candle 不會。

**教訓**：所有 `tensor.transpose(...)?` 後若做 matmul，必須加 `.contiguous()?`。
```rust
// 錯誤：transposed matmul without contiguous
let q = q.transpose(1, 2)?.matmul(&k.transpose(1, 2)?)?;

// 正確
let q = q.transpose(1, 2)?.contiguous()?;
let k = k.transpose(1, 2)?.contiguous()?;
let attn = q.matmul(&k)?;
```
受影響模組：`attention.rs`（GQA）、`ralm.rs`（forward_no_rope）、
`feat_encoder.rs`、`tslm.rs`（tied_weight transpose）。

## CUDA 編譯環境（2026-06-27，2026-06-29 更新）

**現象**：`cargo build --features cuda` 需要 MSVC `cl.exe` 在 PATH 中。

**原因**：candle-kernels v0.9.2 的 build script (`bindgen_cuda v0.1.6`) 在 CUDA 編譯期間
會呼叫 MSVC 的 cl.exe 進行 host 編譯。

**解法**（二選一）：

1. **完整解法**：先執行 `vcvars64.bat` 設定 Visual Studio 環境變數，再執行 cargo。
```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
cargo build --features cuda
```

2. **一行參數解法**：設定 `NVCC_CCBIN` 環境變數（bindgen_cuda 原生支援）。
   `NVCC_CCBIN` 會被 bindgen_cuda 傳遞給 nvcc 的 `-ccbin` 參數，
   同時自動加 `-allow-unsupported-compiler` 標誌。
   需搭配已設定的 MSVC INCLUDE/LIB 環境（由 vcvars64.bat 提供）。
```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
$env:NVCC_CCBIN = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64"
cargo build --features cuda
```

已在 `.cargo/config.toml` 的 `[env]` 區塊設定 `NVCC_CCBIN`，
執行 cargo 時會自動啟用。

## MSVC 路徑（2026-06-27）

VS 2022 Community 安裝於非預設路徑（`C:\Program Files\Microsoft Visual Studio\18\Community`，
而非常見的 `\17\`）。cl.exe 實際路徑：
- `...\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe`（vcvars64.bat 啟用的版本）
- `...\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe`（較舊版）

確認 cl.exe 位置：`where.exe cl.exe`

---
## Lesson #4 — 2026-06-28
**Trigger:** VoxCPM2 Rust CUDA 已能產生清楚人聲，但仍有些微雜音。
**Rule:** 在寫出 generated speech 前，先量測 DC/peak/high-frequency energy；若 raw AudioVAE waveform 仍有 DC offset、近滿刻度峰值或 12kHz 以上殘留，套用保守輸出閘（DC removal、light low-pass、edge fade、headroom limiter）再寫 PCM，並用 ASR 確認人聲仍可辨識。
**Source:** reduce residual speech noise

## Lesson #5 — 2026-06-28 (Critical Bug)
**Trigger:** CFM CFG unconditional path used `cond=zeros` instead of `cond=cond`, causing 42.9% low-frequency rumble.
**Rule:** When implementing CFG zero-star CFM, the unconditional path must receive the SAME `cond` (audio prefix) as the conditional path — only `mu` (text condition) should differ. Python reference: `cond_in[:b], cond_in[b:] = cond, cond`. Using zeros for `cond` in the unconditional path breaks the classifier-free guidance steering.
**Source:** `unified_cfm.rs:lines 190-192` — replaced `cond_null=zeros` with `cond_2x = cat(&[cond, cond])`.

## Lesson #6 — 2026-06-28
**Trigger:** Prefill h_lm/h_res matched perfectly between Python and Rust (cos_sim > 0.9999), but CFM pred_feat diverged completely (cos_sim ~0.05) even after fixing the cond bug.
**Rule:** CFM noise generation (`make_randn` Box-Muller vs `torch.randn`) produces completely different noise tensors even with the same seed. This is an expected RNG implementation difference — Box-Muller (Rust) vs Philox/Ziggurat (PyTorch CUDA) are fundamentally different algorithms. The resulting trajectories diverge from the first step. For perfect reproducibility, pre-generate noise in Python and load as a static tensor in Rust during comparison runs.
**Source:** `unified_cfm.rs:make_randn` vs `unified_cfm.py:torch.randn`

## Lesson #7 — 2026-06-29
**Trigger:** After CFM cond/latent fixes and 12kHz low-pass, generated speech was intelligible but still had audible residual noise.
**Rule:** When 12kHz+ hiss is already low, measure sub-bass and quiet-frame RMS before adding more low-pass. If 0-80Hz or quiet-frame RMS remains elevated, add a conservative high-pass and soft expander; do not lower CFG blindly because it can reduce high-frequency noise while worsening text intelligibility.
**Source:** third residual noise reduction pass

---
## Lesson #8 — 2026-06-29
**Trigger:** Mandarin speech still sounded noisy when the prompt used Traditional Chinese.
**Rule:** For Mandarin quality checks, use Simplified Chinese text and seed 99 first; Traditional Chinese characters can bias VoxCPM2 toward Cantonese, so validate the intended language with ASR before changing audio filters.
**Source:** Mandarin residual noise pass
