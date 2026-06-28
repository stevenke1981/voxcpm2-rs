# Acceptance Report — VoxCPM2 Rust/Candle Pipeline

## 驗收狀態

### G0: Workspace scaffold (`cargo test --workspace --features cpu`)

**Status: ✅ PASS**
```
23 passed; 0 failed; 10 ignored
```
All 23 non-ignored unit tests pass on CPU.

### G1: Model inspect

**Status: ✅ PASS**
- `model.safetensors` (577 tensors, BF16) ✓
- `audiovae.safetensors` (312 tensors, F32, weight_norm fused) ✓
- `config.json`, `tokenizer.json`, `special_tokens_map.json` ✓
- CLI `voxcpm2 inspect` works ✓

### G2: Tokenizer parity

**Status: ✅ PASS** (14/14 alignment confirmed in earlier milestone)

### G3: Tensor load

**Status: ✅ PASS**
- `load_main_vb()` loads all 577 tensors ✓
- `load_audiovae_vb()` loads all 312 tensors ✓
- Weight name prefix mapping verified ✓

### G4: Submodule forward shapes

**Status: ✅ PASS** (all shape tests pass)
- TSLM: `[B, T, 2048]` ✓
- RALM: `[B, T, 2048]` ✓
- LocEnc: `[B, 1, 2048]` ✓
- LocDiT: `[B, 64, 4]` ✓
- AudioVAE: `[B, 1, samples]` ✓
- FSQ: `[B, T, 2048]` ✓
- StopHead: `[B, T, 2]` ✓
- FeatEncoder: `[B, S, 1024]` ✓
- UnifiedCFM Euler solver: ✓

### G5: Audio smoke

**Status: ✅ PASS**
- Dry-run produces 48kHz WAV with valid sine tone
- Real pipeline produces valid WAV on CUDA: 122880 samples @ 48kHz (~2.56s)
- AudioVAE decode on CUDA confirmed working (conv1d/convtranspose1d all CUDA-compatible)

### G6: Quality parity

**Status: ✅ SIGNIFICANTLY IMPROVED — Listenable speech with 79.4% speech energy**

#### Latest results (cond fix applied 2026-06-28)

| Metric | Before (broken cond) | After (fixed cond) | Improvement |
|--------|---------------------|-------------------|-------------|
| Speech energy (300-8000 Hz) | 55.5% (best seed) | **79.4%** (cfg=2.5) | **+23.9pp** |
| Low-freq rumble (100-300 Hz) | 42.9% | **14.8%** | **-28.1pp** |
| AudioVAE model.7 peak | 42-97 (OOD) | **12.5** (in-distribution) | Safe |
| AudioVAE model.8 peak | ~72 (OOD) | **16.1** (in-distribution) | Safe |
| Duration | ~3.84s | 3.84s | Same |
| AR steps | 23 steps (stop at 22) | 24 steps (stop at 23) | Minor |

#### What was fixed
- **CRITICAL BUG**: CFM classifier-free guidance unconditional path used `cond=zeros` instead of `cond=cond`. This caused the CFG steering to operate on incorrect unconditional predictions, producing latents far from the training distribution. Fix: `unified_cfm.rs` lines 190-192 — replaced `cond_null = zeros` with `cond_2x = cat(&[cond, cond])` matching Python's `cond_in[:b], cond_in[b:] = cond, cond`.

#### Best configurations (seed=100)

| cfg | Speech Energy | Rumble | Notes |
|-----|--------------|--------|-------|
| 1.0 | 70.1% | 22.5% | Zero-star CFG only (no CFG) |
| 2.0 | 76.4% | 16.7% | Default |
| **2.5** | **79.4%** | **14.8%** | **Best** |
| 3.0 | 78.9% | 14.3% | Slightly cleaner high-end |
| 3.5 | 67.5% | 23.3% | Over-steering |
| 4.0 | 69.2% | 21.7% | Over-steering |

#### CFG sweep analysis
- cfg=2.5 gives the best balance of speech clarity and low rumble
- cfg>3.0 causes over-steering (speech energy drops, rumble increases)
- Even cfg=1.0 (zero-star CFG only) is better than the old broken cfg=2.0 (70.1% vs 55.5%)

#### Remaining issues
- 🟡 **RNG difference**: Box-Muller (Rust) vs `torch.randn` (Python) — completely different RNG algorithms even with same seed. Trajectories diverge from step 0. This is expected — the same text SHOULD produce different audio with different random noise.
- 🟡 **Latent std**: Rust ~1.60 vs Python ~1.26 — still higher but AudioVAE model.7 peak 12.5 is in-distribution, producing clean audio.
- FSQ tanh/round order bug: **FIXED 2026-06-28** (was `round→tanh`, now `tanh→round`)
- `max_len` formula aligned with Python: `min(seq_len * 6 + 10, global_max)` — 16 steps for 1 token
- Stop head matches Python: `Linear(2048,2048) + SiLU + Linear(2048,2,no_bias)` + argmax
- **CFM seed propagation FIXED 2026-06-28**: `make_randn` now accepts `seed: Option<u64>` with `StdRng` + Box-Muller for reproducible CUDA noise across runs.

### G7: egui GUI

**Status: ✅ PASS** (scaffold + tabs)
- Model tab: directory browser, device selector, inspect ✓
- Synthesis tab: text input, CFG, steps, seed, dry-run ✓
- Output tab: waveform, rodio playback, WAV export ✓
- Diagnostics tab: device info, VRAM estimate, event log ✓
- Cancel generation ✓
- Cloning tab: UI scaffold (disabled backend) ✓

### G8: GPU benchmark

**Status: ✅ PASS** (CUDA end-to-end inference confirmed)

- `cargo build --features cuda` ✅ (successful, 50.52s incl. CUDA kernel compilation)
- `cargo test --features cuda --lib` ✅ (23 passed, 0 failed, 10 ignored)
- CUDA end-to-end inference with AudioVAE on CUDA ✅
- Output: 2.56s 48kHz WAV (valid RIFF, 16-bit mono PCM)
- Full pipeline test PASSED in 58.4s (including CUDA compilation in test build)
- GPU: RTX 3060 Ti, device ID 1, VRAM 5244 MiB during autoregressive inference
- CUDA fix: `NVCC_CCBIN` env var (bindgen_cuda passes `-ccbin` to nvcc + `-allow-unsupported-compiler`)
- Persistent config: `.cargo/config.toml` `[env]` section now sets `NVCC_CCBIN` automatically

**One-line CUDA build workflow:**
```powershell
# Set up MSVC environment (one time per shell)
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
# NVCC_CCBIN is auto-configured via .cargo/config.toml
cargo build --features cuda
```

## 已修復的 Python 對齊問題

| 修復 | Python | Rust (修復後) | 日期 |
|---|---|---|---|---|
| `prefix_feat_cond` 初始形狀 | `[1, 64, 4]` 全零 | `[1, feat_dim, patch_size]` 全零 | 2026-06-28 |
| `enc_to_lm_proj` bias | 有 bias | `linear(...)` 含 bias | 2026-06-28 |
| CUDA build `NVCC_CCBIN` | N/A | bindgen_cuda 原生支援 `-ccbin` | 2026-06-29 |
| `max_len` 硬編碼 500 | `min(seq_len*6+10, 2000)` | `min(seq_len*6+10, global_max)` | 2026-06-29 |
| AudioVAE CPU-only | CUDA (PyTorch) | CUDA (candle, conv1d/convtranspose1d 可執行) | 2026-06-29 |
| **FSQ tanh/round 順序** | `in_proj → tanh → round → out_proj` | `in_proj → tanh → round → out_proj` (🐛 原是 `round → tanh`) | **2026-06-28** |
| **CFM seed propagation** | `torch.randn(seed=s)` | `StdRng` + Box-Muller with `seed: Option<u64>` | **2026-06-28** |
| **🔴 CFM CFG uncond cond bug** | `cond_in[:b], cond_in[b:] = cond, cond` | `cond_2x = cat(&[cond, cond])` (🐛 原是用 zeros) | **2026-06-28** |

## 已知限制

1. CPU 推理非常慢（~4.6GB 模型權重，28層 TSLM + 8層 RALM）
2. CUDA 需要 Visual Studio 2022 Community（或 Build Tools）提供 cl.exe
3. 需先執行 `vcvars64.bat` 設定 MSVC INCLUDE/LIB 環境變數（NVCC_CCBIN 已自動設定）
4. 生成的音訊在短文本 + 少步數時 peak 很低（需 post_gain 或更多步數）
5. FeatEncoder vs LocEnc 有兩個獨立編碼器（FeatEncoder 死代碼，可清理）
6. AudioVAE CUDA decode 速度仍需改善（~30s 含模型載入 + 16-step autoregressive + 960x upsampling）
7. Stop head 在短文本時可能不觸發（與 Python 行為一致，因 max_len 動態裁減使問題不明顯）
8. 📌 CFM Box-Muller RNG 與 PyTorch `torch.randn` 天生不同——同 seed 會產生不同噪聲、不同軌跡
9. 📌 Rust latent std (~1.60) 仍高於 Python (~1.26)，但 AudioVAE 解碼器能正常處理（model.7/model.8 峰值 12-16，在分布內）
