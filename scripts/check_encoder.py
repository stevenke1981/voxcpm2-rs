"""Check if AudioVAE encoder-decoder works end-to-end."""
import json
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

tensors = load_file("models/VoxCPM2/audiovae.safetensors")

# Fuse weight_norm
fused = {}
for k in list(tensors.keys()):
    if k.endswith(".weight_g"):
        base = k[: -len(".weight_g")]
        vk = base + ".weight_v"
        if vk in tensors:
            wg = tensors[k].squeeze()
            wv = tensors[vk]
            norm = torch.linalg.vector_norm(wv.reshape(wv.shape[0], -1), dim=1)
            wg_r = wg.reshape(1, 1, 1) if wg.ndim == 0 else wg[:, None, None]
            fused[base + ".weight"] = wg_r * (wv / norm[:, None, None])

# Load config
with open("models/VoxCPM2/config.json") as f:
    cfg = json.load(f)

enc_rates = cfg["audio_vae_config"]["encoder_rates"]
dec_rates = cfg["audio_vae_config"]["decoder_rates"]
latent_dim = cfg["audio_vae_config"]["latent_dim"]
enc_dim = cfg["audio_vae_config"]["encoder_dim"]
dec_dim = cfg["audio_vae_config"]["decoder_dim"]
sample_rate = cfg["audio_vae_config"]["sample_rate"]

print(f"Encoder rates: {enc_rates}")
print(f"Decoder rates: {dec_rates}")
print(f"Latent dim: {latent_dim}, Enc dim: {enc_dim}, Dec dim: {dec_dim}")
print(f"Sample rate: {sample_rate}")

# Check encoder tensor shapes to understand architecture
print("\n=== ENCODER TENSORS ===")
for k in sorted(tensors.keys()):
    if k.startswith("encoder."):
        print(f"  {k}: {list(tensors[k].shape)}")

print("\n=== DECODER TENSORS (model structure) ===")
for k in sorted(tensors.keys()):
    if k.startswith("decoder.model.") and not k.endswith(".weight_v") and not k.endswith(".weight_g"):
        print(f"  {k}: {list(tensors[k].shape)}")
