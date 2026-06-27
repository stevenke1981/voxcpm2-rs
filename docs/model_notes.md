# model_notes.md — VoxCPM2 來源重點

來源：

- Hugging Face model card: https://huggingface.co/openbmb/VoxCPM2
- GitHub repo: https://github.com/OpenBMB/VoxCPM
- Docs: https://voxcpm.readthedocs.io/
- Candle docs: https://huggingface.github.io/candle/
- egui repo: https://github.com/emilk/egui

## 模型事實摘要

- Model ID：`openbmb/VoxCPM2`
- 任務：Text-to-Speech / multilingual / voice-cloning / voice-design
- 權重：`model.safetensors` 約 4.58GB；`audiovae.pth` 約 377MB
- 參數量：約 2B
- dtype：BF16
- 官方建議 VRAM：約 8GB
- 架構：Tokenizer-free Diffusion Autoregressive，主要管線可視為 `LocEnc → TSLM → RALM → LocDiT → AudioVAE V2`
- 輸入/輸出：可接受 16kHz reference，AudioVAE V2 輸出 48kHz
- 支援語言：官方標示 30 種語言，含中文、英文、日文、韓文等
- 官方 Python 需求：Python >= 3.10、PyTorch >= 2.5、CUDA >= 12.0

## config.json 需要實作的欄位

```text
lm_config.hidden_size = 2048
lm_config.intermediate_size = 6144
lm_config.num_hidden_layers = 28
lm_config.num_attention_heads = 16
lm_config.num_key_value_heads = 2
lm_config.vocab_size = 73448
encoder_config.hidden_dim = 1024
encoder_config.num_layers = 12
dit_config.hidden_dim = 1024
dit_config.num_layers = 12
audio_vae_config.latent_dim = 64
audio_vae_config.sample_rate = 16000
audio_vae_config.out_sample_rate = 48000
max_length = 8192
dtype = bfloat16
```

## Rust/Candle 風險

- Candle 已支援 safetensors 與 GPU backend，但 VoxCPM2 的完整模型層需要自行 port。
- `audiovae.pth` 是 PyTorch pickle 格式，Rust 端不應直接反序列化不可信 pickle；應用 Python + torch 在可信環境中載入後轉 safetensors。
- `tokenization_voxcpm2.py` 有自訂 tokenizer 行為，Rust 端需比對 Python tokenizer 的 token IDs，特別是中文 multi-char split 行為。
