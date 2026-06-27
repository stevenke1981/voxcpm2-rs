# gpu.md — GPU 加速設計

## Feature flags

```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]
mkl = ["candle-core/mkl"]
```

## Device policy

`--device auto`：

1. 若 binary 有 `cuda` feature 且 CUDA 初始化成功 → `cuda:0`
2. 否則若有 `metal` feature 且 Metal 可用 → `metal:0`
3. 否則 → `cpu`

## Memory policy

- 預設用 BF16 載入主模型。
- 若 backend 不支援 BF16，嘗試 FP16；最後 fallback FP32。
- KV cache 需可估算記憶體：

```text
layers * seq_len * kv_heads * head_dim * 2(K,V) * dtype_bytes
```

以 config：28 layers、2 kv heads、128 head_dim 為估算基礎。

## CLI benchmark

```bash
voxcpm2 benchmark --model-dir models/VoxCPM2 --device cuda --steps 10 --repeat 3
```

記錄：

- load time
- synth time
- audio seconds
- RTF
- peak VRAM/RAM
- output wav path

## Windows CUDA 注意

- 需要 NVIDIA driver 與 CUDA runtime。
- Candle 的 CUDA feature 可能需要對應 CUDA toolkit / nvcc。
- 若 build 失敗，先用 CPU feature 確認非模型邏輯問題。

## Metal 注意

- Metal 主要針對 macOS Apple Silicon。
- Windows 不使用 Metal。

## OOM 策略

CLI：

```bash
--max-len 256 --steps 6 --no-stream-cache
```

GUI：

- 顯示「降低 steps / 縮短文字 / 關閉 clone / 改 CPU offload」建議。
- generation worker 捕捉 OOM，不讓 GUI crash。
