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

### 未變更的可回滾路徑

- 所有 PyTorch golden 測試保留在 `tests/golden/`
- Python 參考實作保留在 `D:\VoxCPM\`
