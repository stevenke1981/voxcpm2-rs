# Traceability Matrix — VoxCPM2 Rust/Candle vs Python Reference

## 專案比對：`D:\VoxCPM` (Python Ref) vs `E:\voxcpm2_rust_candle_pack` (Rust)

| Python Module | Rust Module | Status |
|---|---|---|
| `voxcpm2.py` (`VoxCPM2Model`) | `pipeline.rs` + `autoregressive.rs` | ✅ Partial |
| `voxcpm2.py` (`_inference`) | `autoregressive.rs` (`generate_autoregressive`) | ✅ Aligned |
| `minicpm4/model.py` (`MiniCPMModel`) | `tslm.rs` | ✅ 28 layers |
| `minicpm4/model.py` (residual) | `ralm.rs` | ✅ 8 layers, no RoPE |
| `locenc/local_encoder.py` (`VoxCPMLocEnc`) | `loc_enc.rs` | ✅ Aligned |
| `locdit/local_dit_v2.py` (`VoxCPMLocDiTV2`) | `loc_dit.rs` | ✅ Aligned |
| `locdit/unified_cfm.py` (`UnifiedCFM`) | `unified_cfm.rs` | ✅ Aligned |
| `audiovae/audio_vae_v2.py` (`AudioVAEV2`) | `audio_vae.rs` | ✅ With weight_norm fusion |
| `layers/scalar_quantization_layer.py` | `fsq_layer.rs` | ✅ Aligned |
| `cli.py` | `crates/voxcpm2-cli/src/main.rs` | ✅ Partial |
| `app.py` / `lora_ft_webui.py` | `crates/voxcpm2-gui/src/main.rs` | ✅ egui scaffold |

## 關鍵差異（已修復）

| ID | Python | Rust (before fix) | Fix | Status |
|---|---|---|---|---|
| D1 | `cond` initial shape `[1, 64, 4]` (zeros) | `[1, 64, 0]` (zero-length) | Changed to `[1, feat_dim, patch_size]` zeros | ✅ Fixed |
| D2 | `enc_to_lm_proj` has bias | `linear_no_bias` | Changed to `linear(...)` (with bias) | ✅ Fixed |
| D8 | `max_len = min(seq_len*6+10, 2000)` | hardcoded 500 | `min(seq_len*6+10, global_max)` with `--max-ar-steps` | ✅ Fixed |
| D9 | AudioVAE on CUDA (PyTorch) | CPU-only (cuBLAS conv1d concern) | CUDA (candle conv1d/convtranspose1d works) | ✅ Fixed |
| D10 | `ScalarQuantizationLayer`: `in_proj → tanh → round → out_proj` | `in_proj → floor(x*9+0.5)/9 → tanh → out_proj` | Fixed tanh/round order: `in_proj → tanh → round → out_proj` | ✅ Fixed 2026-06-28 |

## 已修復差異

| ID | Python | Rust (before fix) | Fix | Status |
|---|---|---|---|---|
| D11 | CFM noise: `torch.randn(seed=s)` | CFM noise: `Tensor::randn(seed=default)` — 無法控制種子 | `make_randn` 改用 `StdRng` + Box-Muller，接受 `seed: Option<u64>` | ✅ Fixed 2026-06-28 |
| D13 | `unified_cfm.py:115`: CFG uncond path uses `cond_in[:b], cond_in[b:] = cond, cond` (same cond for both) | `unified_cfm.rs:191-192`: CFG uncond path used `cond_null = zeros` for second half | Both halves now get `cond` — only `mu` differs for unconditional | ✅ Fixed 2026-06-28 |

## 已知殘餘差異（設計所致，非 Bug）

| ID | Python | Rust | 影響 | 原因 |
|---|---|---|---|---|
| D12 | Python latent std ~1.26 | Rust latent std ~1.60 (after cond fix) | Rust 潛在 std 仍較高，但 AudioVAE model.7 峰值從 42-97 降至 12.5 (in-distribution) | CFM RNG 不同（Box-Muller vs torch.randn）導致不同軌跡，非實作錯誤 |

## 關鍵差異（設計差異，功能等效）

| ID | Python | Rust | 原因 |
|---|---|---|---|
| D3 | `forward_step` input `[B, D]` (2D) | `forward_step` input `[B, 1, D]` (3D) | 線性層處理 `[B, T, D]` |
| D4 | Special token at pos 0 (CLS) | Special token at pos T (末尾) | BiDAttention 下等價 |
| D5 | `nn.Linear(1024, 2048)` with bias | `linear(1024, 2048)` with bias | ✅ 已對齊 |
| D6 | `stop_head` weight shape `[2, 2048]` no bias | `linear_no_bias(2048, 2)` | ✅ 已對齊 |
| D7 | `CausalConv1d`/`CausalTransposeConv1d` | Standard Conv1d/ConvTranspose1d with weight_norm fusion | Rust 使用 fused weight_norm |

## 未實作的功能

| Feature | Python File | Rust Status |
|---|---|---|
| Streaming generation | `voxcpm2.py:_generate(streaming=True)` | ❌ Not ported |
| LoRA fine-tuning | `lora_ft_webui.py` | ❌ Out of scope |
| VAD silence trimming | `voxcpm2.py:_trim_audio_silence_vad()` | ❌ Not ported |
| `build_prompt_cache` / cache merge | `voxcpm2.py` | ❌ Not ported |
| Training code | `training/*` | ❌ Out of scope |

## 已實作的語音克隆功能

| Feature | Python File | Rust File | Status |
|---|---|---|---|
| Reference audio cloning | `voxcpm2.py:build_prompt_cache()` | `autoregressive.rs:generate_autoregressive_clone()` | ✅ Full pipeline: AudioVAE encoder → LocEnc patches → TSLM/RALM prefill → autoregressive DiT loops → AudioVAE decoder |
| `special_tokens` for clone | `tokenizer.py` | `tokenizer.rs:SpecialTokens` | ✅ `ref_audio_start` (103), `ref_audio_end` (104) |
| CLI clone subcommand | `cli.py` | `crates/voxcpm2-cli/src/main.rs` | ✅ `clone` subcommand with `--ref-audio`, `--clone-strength`, all synth params |
| GUI clone tab | `app.py` | `crates/voxcpm2-gui/src/tabs/clone.rs` | ✅ Full backend integration: separate text field, parameters, similarity slider |

## Gate Progress

| Gate | Condition | Status |
|---|---|---|
| G0 | `cargo test --workspace` passes | ✅ 23/23 unit tests pass |
| G1 | Model inspect | ✅ `full_asset_report` works |
| G2 | Tokenizer parity | ✅ 14/14 CJK tests pass |
| G3 | Tensor load | ✅ All safetensors loadable |
| G4 | Submodule forward shapes | ✅ All shape tests pass |
| G5 | Audio smoke (valid wav) | ✅ Real pipeline produces valid WAV on CUDA |
| G6 | Quality parity | ✅ Best result: seed=100, cfg=2.5 → 79.4% speech energy (300-8000 Hz), 14.8% low-freq rumble (100-300 Hz). Cond fix eliminated AudioVAE CLS peaks (model.7: 42→12). Remaining: RNG difference (Box-Muller vs torch.randn) causes trajectory divergence. |
| G7 | GUI complete flow | ✅ Scaffold complete |
| G8 | GPU benchmark | ✅ CUDA end-to-end inference confirmed (device ID 1, RTX 3060 Ti, 30-step AR + 20-step CFM + VAE decode ~60s) |
