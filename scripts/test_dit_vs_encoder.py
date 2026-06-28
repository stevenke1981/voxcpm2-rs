"""Compare DiT latent distribution vs AudioVAE encoder latent distribution."""
import json, wave, array, struct
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np

# Load model
tensors = load_file("models/VoxCPM2/audiovae.safetensors")
with open("models/VoxCPM2/config.json") as f:
    cfg = json.load(f)

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

# ── ENCODER: encode audio to latent ──
def encode(audio):
    """audio: [1, 1, T] at 16000 Hz"""
    h = F.conv1d(audio, fused["encoder.block.0.weight"], tensors["encoder.block.0.bias"], padding=3)
    h = F.silu(h)
    
    # Block 1: rate=2 (k=4)  
    # Sub-blocks 0,1,2 (indices 0,1,2 for the 3 residual sub-blocks within block.1)
    for si in range(3):  # sub-block index 0,1,2
        res = h
        h = h * tensors[f"encoder.block.1.block.{si}.block.0.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.1.block.{si}.block.1.weight"], tensors[f"encoder.block.1.block.{si}.block.1.bias"], padding=3, groups=128))
        h = h * tensors[f"encoder.block.1.block.{si}.block.2.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.1.block.{si}.block.3.weight"], tensors[f"encoder.block.1.block.{si}.block.3.bias"], padding=0))
        h = F.silu(h + res)
    h = h * tensors["encoder.block.1.block.3.alpha"]  # alpha after sub-blocks
    h = F.conv1d(h, fused["encoder.block.1.block.4.weight"], tensors["encoder.block.1.block.4.bias"], stride=2, padding=1)
    h = F.silu(h)  # [1, 256, T/2]
    
    # Block 2: rate=5 (k=10)
    for si in range(3):
        res = h
        h = h * tensors[f"encoder.block.2.block.{si}.block.0.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.2.block.{si}.block.1.weight"], tensors[f"encoder.block.2.block.{si}.block.1.bias"], padding=3, groups=256))
        h = h * tensors[f"encoder.block.2.block.{si}.block.2.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.2.block.{si}.block.3.weight"], tensors[f"encoder.block.2.block.{si}.block.3.bias"], padding=0))
        h = F.silu(h + res)
    h = h * tensors["encoder.block.2.block.3.alpha"]
    h = F.conv1d(h, fused["encoder.block.2.block.4.weight"], tensors["encoder.block.2.block.4.bias"], stride=5, padding=2)
    h = F.silu(h)  # [1, 512, T/10]
    
    # Block 3: rate=8 (k=16)
    for si in range(3):
        res = h
        h = h * tensors[f"encoder.block.3.block.{si}.block.0.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.3.block.{si}.block.1.weight"], tensors[f"encoder.block.3.block.{si}.block.1.bias"], padding=3, groups=512))
        h = h * tensors[f"encoder.block.3.block.{si}.block.2.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.3.block.{si}.block.3.weight"], tensors[f"encoder.block.3.block.{si}.block.3.bias"], padding=0))
        h = F.silu(h + res)
    h = h * tensors["encoder.block.3.block.3.alpha"]
    h = F.conv1d(h, fused["encoder.block.3.block.4.weight"], tensors["encoder.block.3.block.4.bias"], stride=8, padding=4)
    h = F.silu(h)  # [1, 1024, T/80]
    
    # Block 4: rate=8 (k=16)
    for si in range(3):
        res = h
        h = h * tensors[f"encoder.block.4.block.{si}.block.0.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.4.block.{si}.block.1.weight"], tensors[f"encoder.block.4.block.{si}.block.1.bias"], padding=3, groups=1024))
        h = h * tensors[f"encoder.block.4.block.{si}.block.2.alpha"]
        h = F.silu(F.conv1d(h, fused[f"encoder.block.4.block.{si}.block.3.weight"], tensors[f"encoder.block.4.block.{si}.block.3.bias"], padding=0))
        h = F.silu(h + res)
    h = h * tensors["encoder.block.4.block.3.alpha"]
    h = F.conv1d(h, fused["encoder.block.4.block.4.weight"], tensors["encoder.block.4.block.4.bias"], stride=8, padding=4)
    h = F.silu(h)  # [1, 2048, T/640]
    
    # VAE: mu, logvar
    mu = F.conv1d(h, fused["encoder.fc_mu.weight"], tensors["encoder.fc_mu.bias"], padding=1)
    logvar = F.conv1d(h, fused["encoder.fc_logvar.weight"], tensors["encoder.fc_logvar.bias"], padding=1)
    # Reparameterize
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # [1, 64, T/640]
    return z, mu, logvar

# ── Test with a generated sine tone ──
print("=== Encode sine tone ===")
t = torch.arange(0, 48000) / 16000
audio = 0.5 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0).unsqueeze(0)  # [1, 1, 48000]
z, mu, logvar = encode(audio)
print(f"Latent z: shape={list(z.shape)} mean={z.mean():.6f} std={z.std():.6f} min={z.min():.6f} max={z.max():.6f}")
print(f"Latent mu: mean={mu.mean():.6f} std={mu.std():.6f} min={mu.min():.6f} max={mu.max():.6f}")
print(f"Latent logvar: mean={logvar.mean():.6f} std={logvar.std():.6f}")

# Compare with random noise latent
z_rand = torch.randn_like(z)
print(f"\nRandom latent: mean={z_rand.mean():.6f} std={z_rand.std():.6f} min={z_rand.min():.6f} max={z_rand.max():.6f}")

# ── DECODE both ──
def decode(z):
    """z: [1, 64, T]"""
    h = F.silu(F.conv1d(z, fused["decoder.model.0.weight"], tensors["decoder.model.0.bias"], padding=3, groups=64))
    h = F.silu(F.conv1d(h, fused["decoder.model.1.weight"], tensors["decoder.model.1.bias"]))
    
    specs = {2:(16,8,2048,1024),3:(12,6,1024,512),4:(10,5,512,256),5:(4,2,256,128),6:(4,2,128,64),7:(4,2,64,32)}
    for n in range(2, 8):
        k,s,ci,co = specs[n]
        sk = f"decoder.sr_cond_model.{n}.scale_embed.weight"
        bk = f"decoder.sr_cond_model.{n}.bias_embed.weight"
        h = h * tensors[sk][3:4,:,None] + tensors[bk][3:4,:,None]
        h = h * tensors[f"decoder.model.{n}.block.0.alpha"]
        pad = max(0, (k - s + 1) // 2) if k > s else 0
        op = s - k + 2 * pad
        h = F.conv_transpose1d(h, fused[f"decoder.model.{n}.block.1.weight"], tensors[f"decoder.model.{n}.block.1.bias"], stride=s, padding=pad, output_padding=op)
        h = F.silu(h)
        for i in range(2, 5):
            res = h
            h = h * tensors[f"decoder.model.{n}.block.{i}.block.0.alpha"]
            h = F.silu(F.conv1d(h, fused[f"decoder.model.{n}.block.{i}.block.1.weight"], tensors[f"decoder.model.{n}.block.{i}.block.1.bias"], padding=3, groups=co))
            h = h * tensors[f"decoder.model.{n}.block.{i}.block.2.alpha"]
            h = F.silu(F.conv1d(h, fused[f"decoder.model.{n}.block.{i}.block.3.weight"], tensors[f"decoder.model.{n}.block.{i}.block.3.bias"], padding=0))
            h = F.silu(h + res)
    
    h = h * tensors["decoder.model.8.alpha"]
    out = F.conv1d(h, fused["decoder.model.9.weight"], tensors["decoder.model.9.bias"], padding=3)
    return out

print("\n=== Decode comparison ===")
out_enc = decode(z)
out_rand = decode(z_rand)

print(f"Encoded audio decoded: mean={out_enc.mean():.6f} std={out_enc.std():.6f} min={out_enc.min():.6f} max={out_enc.max():.6f}")
print(f"Random latent decoded: mean={out_rand.mean():.6f} std={out_rand.std():.6f} min={out_rand.min():.6f} max={out_rand.max():.6f}")
print(f"  First 10 encoded: {out_enc[0,0,:10].tolist()}")
print(f"  First 10 random:  {out_rand[0,0,:10].tolist()}")

# Save encoded audio
out_enc_np = out_enc[0,0].detach().numpy()
peak = np.max(np.abs(out_enc_np))
print(f"\nEncoded-then-decoded peak: {peak:.6f}")

# Normalize and save
if peak > 1e-8:
    out_enc_np = out_enc_np / peak * 0.9  # normalize to 0.9 FS
    with wave.open("output/encode_decode.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.int16(out_enc_np * 32767).tobytes())
    print(f"Saved output/encode_decode.wav ({len(out_enc_np)} samples)")
else:
    print("Output is near-zero, not saving")
