# conversion.md — 模型下載與轉換策略

## 檔案來源

官方 HF repo 主要檔案：

```text
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
tokenization_voxcpm2.py
model.safetensors
audiovae.pth
```

## 為什麼要轉 `audiovae.pth`

`audiovae.pth` 是 PyTorch pickle 權重。Rust/Candle 適合讀 safetensors；pickle 不適合在 Rust 端直接解析，也有安全風險。建議流程：

1. 在可信 Python 環境中用 torch 載入 `.pth`。
2. 展平成 `state_dict`。
3. 儲存成 `audiovae.safetensors`。
4. Rust/Candle 只讀 safetensors。

## 轉換命令

```powershell
python scripts/convert_audiovae_pth_to_safetensors.py `
  --input models/VoxCPM2/audiovae.pth `
  --output models/VoxCPM2/audiovae.safetensors `
  --manifest models/VoxCPM2/audiovae_manifest.json
```

## Manifest 格式

```json
{
  "repo": "openbmb/VoxCPM2",
  "files": {
    "model.safetensors": {"sha256": "...", "bytes": 491...},
    "audiovae.safetensors": {"sha256": "...", "bytes": 395...}
  },
  "tensors": {
    "model": [{"name":"...", "shape":[...], "dtype":"BF16"}],
    "audiovae": [{"name":"...", "shape":[...], "dtype":"F32"}]
  }
}
```

## Candle 載入原則

- `model.safetensors`：使用 `candle_core::safetensors::load` 或 `VarBuilder::from_mmaped_safetensors`。
- `audiovae.safetensors`：單獨 namespace 載入，避免與主模型 tensor name 衝突。
- shape 不一致時 fail fast，不進入推理。

## 轉換驗收

- `inspect_safetensors.py` 可列出 tensor。
- Rust `voxcpm2 inspect` 可讀取 manifest。
- 必要 AudioVAE decoder tensor 全部存在。
