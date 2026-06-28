"""Debug AudioVAE decoder output — check if output is truly constant."""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

tensors = load_file("models/VoxCPM2/audiovae.safetensors")

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

import re
# Check alpha values
print("=== Alpha (gating) values ===")
for k in sorted(tensors.keys()):
    if "alpha" in k:
        t = tensors[k]
        if t.numel() <= 64:
            print(f"  {k}: {t.flatten()[:12].tolist()} mean={t.mean():.6f}")
        else:
            print(f"  {k}: shape={list(t.shape)} mean={t.mean():.6f} min={t.min():.6f} max={t.max():.6f}")

print("\n=== Model.9 (final conv) ===")
wv = tensors["decoder.model.9.weight_v"]
wg = tensors["decoder.model.9.weight_g"]
w = fused["decoder.model.9.weight"]
b = tensors["decoder.model.9.bias"]
print(f"  weight_v shape: {list(wv.shape)}")
print(f"  weight_g: {wg.item():.6f}")
print(f"  fused weight stats: mean={w.mean():.6f} std={w.std():.6f} min={w.min():.6f} max={w.max():.6f}")
print(f"  bias: {b.item():.6f}")
print(f"  DC gain (sum of weights): {w.sum().item():.6f}")

# Decode and compare two different inputs
print("\n=== Full decode with random input ===")
for seed in [42, 12345]:
    torch.manual_seed(seed)
    latent = torch.randn(1, 64, 16)

    h = F.conv1d(latent, fused["decoder.model.0.weight"], tensors["decoder.model.0.bias"], padding=3, groups=64)
    h = F.silu(h)
    h = F.conv1d(h, fused["decoder.model.1.weight"], tensors["decoder.model.1.bias"])
    h = F.silu(h)

    for n in range(2, 8):
        k, s, ci, co = {2: (16, 8, 2048, 1024), 3: (12, 6, 1024, 512), 4: (10, 5, 512, 256), 5: (4, 2, 256, 128), 6: (4, 2, 128, 64), 7: (4, 2, 64, 32)}[n]
        sk = f"decoder.sr_cond_model.{n}.scale_embed.weight"
        bk = f"decoder.sr_cond_model.{n}.bias_embed.weight"
        h = h * tensors[sk][3:4, :, None] + tensors[bk][3:4, :, None]
        h = h * tensors[f"decoder.model.{n}.block.0.alpha"]
        pad = max(0, (k - s + 1) // 2) if k > s else 0
        op = s - k + 2 * pad
        h = F.conv_transpose1d(h, fused[f"decoder.model.{n}.block.1.weight"], tensors[f"decoder.model.{n}.block.1.bias"], stride=s, padding=pad, output_padding=op)
        h = F.silu(h)
        for i in range(2, 5):
            res = h
            h = h * tensors[f"decoder.model.{n}.block.{i}.block.0.alpha"]
            h = F.conv1d(h, fused[f"decoder.model.{n}.block.{i}.block.1.weight"], tensors[f"decoder.model.{n}.block.{i}.block.1.bias"], padding=3, groups=co)
            h = F.silu(h)
            h = h * tensors[f"decoder.model.{n}.block.{i}.block.2.alpha"]
            h = F.conv1d(h, fused[f"decoder.model.{n}.block.{i}.block.3.weight"], tensors[f"decoder.model.{n}.block.{i}.block.3.bias"], padding=0)
            h = F.silu(h + res)

    h = h * tensors["decoder.model.8.alpha"]
    out = F.conv1d(h, fused["decoder.model.9.weight"], tensors["decoder.model.9.bias"], padding=3)

    print(f"  seed={seed}: shape={list(out.shape)} mean={out.mean():.6f} std={out.std():.6f} min={out.min():.6f} max={out.max():.6f}")
    print(f"    First 16 samples: {out[0, 0, :16].tolist()}")

# Now check: is the output also constant for a REAL model?
# Compare two latents with different structure
print("\n=== Two very different latents ===")
torch.manual_seed(42)
latent1 = torch.randn(1, 64, 16) * 0.5  # small
latent2 = torch.randn(1, 64, 16) * 10.0  # large

for name, latent in [("small", latent1), ("large", latent2)]:
    h = F.conv1d(latent, fused["decoder.model.0.weight"], tensors["decoder.model.0.bias"], padding=3, groups=64)
    h = F.silu(h)
    h = F.conv1d(h, fused["decoder.model.1.weight"], tensors["decoder.model.1.bias"])
    h = F.silu(h)
    
    print(f"  {name} after model.0+1: mean={h.mean():.6f} std={h.std():.6f} max={h.abs().max():.6f}")
    
    for n in range(2, 8):
        k, s, ci, co = {2: (16, 8, 2048, 1024), 3: (12, 6, 1024, 512), 4: (10, 5, 512, 256), 5: (4, 2, 256, 128), 6: (4, 2, 128, 64), 7: (4, 2, 64, 32)}[n]
        sk = f"decoder.sr_cond_model.{n}.scale_embed.weight"
        bk = f"decoder.sr_cond_model.{n}.bias_embed.weight"
        h = h * tensors[sk][3:4, :, None] + tensors[bk][3:4, :, None]
        h = h * tensors[f"decoder.model.{n}.block.0.alpha"]
        pad = max(0, (k - s + 1) // 2) if k > s else 0
        op = s - k + 2 * pad
        h = F.conv_transpose1d(h, fused[f"decoder.model.{n}.block.1.weight"], tensors[f"decoder.model.{n}.block.1.bias"], stride=s, padding=pad, output_padding=op)
        h = F.silu(h)
        for i in range(2, 5):
            res = h
            h = h * tensors[f"decoder.model.{n}.block.{i}.block.0.alpha"]
            h = F.conv1d(h, fused[f"decoder.model.{n}.block.{i}.block.1.weight"], tensors[f"decoder.model.{n}.block.{i}.block.1.bias"], padding=3, groups=co)
            h = F.silu(h)
            h = h * tensors[f"decoder.model.{n}.block.{i}.block.2.alpha"]
            h = F.conv1d(h, fused[f"decoder.model.{n}.block.{i}.block.3.weight"], tensors[f"decoder.model.{n}.block.{i}.block.3.bias"], padding=0)
            h = F.silu(h + res)
        if n > 3:  # only print later blocks
            pass  # suppress
    
    h = h * tensors["decoder.model.8.alpha"]
    out = F.conv1d(h, fused["decoder.model.9.weight"], tensors["decoder.model.9.bias"], padding=3)
    
    diff = (out[0,0,:] - out[0,0,0].item()).abs().max()
    print(f"  {name} output: f32 mean={out.mean():.6f}, max={out.abs().max():.6f}, max_deviation_from_first_sample={diff:.6f}")
    print(f"    First 10: {out[0,0,:10].tolist()}")
    print(f"    Samples 100-110: {out[0,0,100:110].tolist()}")
