"""Check all weight_g values in AudioVAE decoder."""
import torch
from safetensors.torch import load_file

tensors = load_file("models/VoxCPM2/audiovae.safetensors")

print("=== Decoder weight_g values ===")
total_scale = []
for k in sorted(tensors.keys()):
    if "weight_g" in k and k.startswith("decoder"):
        g = tensors[k]
        g_flat = g.flatten()
        print(f"  {k}: shape={list(g.shape)} mean={g_flat.mean():.6f} min={g_flat.min():.6f} max={g_flat.max():.6f}")
        total_scale.append(g_flat.abs().mean().item())

print(f"\nOverall mean weight_g: {sum(total_scale)/len(total_scale):.6f}")

# Up block ConvTranspose weight_g
print("\n=== Up block ConvTranspose weight_g ===")
for n in range(2, 8):
    k = f"decoder.model.{n}.block.1.weight_g"
    if k in tensors:
        g = tensors[k].flatten()
        vk = k.replace("weight_g", "weight_v")
        v = tensors[vk]
        norm = torch.linalg.vector_norm(v.reshape(v.shape[0], -1), dim=1).mean()
        print(f"  model.{n} transposed conv: G={g.mean():.6f} V_norm={norm:.6f}")

# Final conv analysis
print("\n=== Final conv (model.9) analysis ===")
a8 = tensors["decoder.model.8.alpha"].flatten()
print(f"  model.8 alpha: mean={a8.mean():.6f}")

w9_g = tensors["decoder.model.9.weight_g"].item()
w9_b = tensors["decoder.model.9.bias"].item()
print(f"  model.9 weight_g={w9_g:.6f} bias={w9_b:.6f}")

# Check if there's a scale or gain parameter missing
print(f"\n  Product: w9_g * mean(alpha) = {w9_g * a8.mean():.10f}")

# Check: what's the max value in ALL conv weights?
all_decoder_w = []
for k in tensors:
    if k.startswith("decoder") and ("weight_v" in k or "weight_g" in k or k.endswith(".bias") or k.endswith(".alpha")):
        all_decoder_w.append(tensors[k].flatten().abs().max().item())
print(f"\n  Max absolute value in ALL decoder weight tensors: {max(all_decoder_w):.6f}")

# Check if any weight_g is near zero
for k in sorted(tensors.keys()):
    if "weight_g" in k and k.startswith("decoder"):
        g = tensors[k].flatten()
        if g.abs().min() < 0.01:
            print(f"  WARNING: near-zero weight_g in {k}: {g.abs().min():.10f}")
