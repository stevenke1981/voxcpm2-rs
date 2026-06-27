# tests

- `golden/` 放官方 Python 產生的 tokenizer/audio/hidden-state golden data。
- 無模型時只跑 Rust unit + CLI dry-run。
- 有模型時設定 `VOXCPM2_MODEL_DIR=models/VoxCPM2` 跑 ignored tests。
