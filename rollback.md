# Rollback Notes — VoxCPM2 Rust/Candle

## Active Changes (uncommitted)

針對 Python `D:\VoxCPM` 比對後的修改：

### 2026-06-28: Python 參考實作對齊

**修改檔案：**
1. `crates/voxcpm2-core/src/autoregressive.rs`
   - 變更：`prefix_feat_cond` 初始形狀 `[1, feat_dim, 0]` → `[1, feat_dim, patch_size]`
   - 原因：Python 使用 `feat[:, -1, ...]` 產生 `[B, P, D]`，transpose 後為 `[B, C, patch_size]`
   - Rollback: 改回 `Tensor::zeros(&[1, feat_dim, 0], ...)`

2. `crates/voxcpm2-core/src/models/loc_enc.rs`
   - 變更：`enc_to_lm_proj` 從 `linear_no_bias` → `linear` (含 bias)
   - 原因：safetensors 中有 `enc_to_lm_proj.bias`
   - Rollback: 改回 `linear_no_bias`

### 2026-06-28: CFM CFG unconditional cond fix

**修改檔案：**
1. `crates/voxcpm2-core/src/models/unified_cfm.rs`
   - 變更：`cond_2x` 從 `cat(&[cond, &cond_null])` (cond_null=zeros) → `cat(&[cond, cond])`
   - 原因：Python 的 CFG 無條件路徑使用相同的 cond（音頻前綴），僅 mu（文字條件）不同
   - Python 參考：`unified_cfm.py:115` — `cond_in[:b], cond_in[b:] = cond, cond`
   - 使用 zero cond 導致 CFG steering 操作在錯誤的無條件預測上，產生分布之外的 latent
   - 影響：model.7 peak 從 42-97 降至 12.5，語音能量從 55.5% 提升至 79.4%
   - Rollback: 改回 `let cond_null = self.make_full(0.0f32, ...); let cond_2x = Tensor::cat(&[cond, &cond_null], 0)?;`

### 未變更的可回滾路徑

- 所有 PyTorch golden 測試保留在 `tests/golden/`
- Python 參考實作保留在 `D:\VoxCPM\`
