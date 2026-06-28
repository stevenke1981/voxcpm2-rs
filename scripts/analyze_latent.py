"""Analyze latent structure."""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else r"E:\voxcpm2_rust_candle_pack\output\latent_rust.f32"
latent = np.fromfile(path, dtype=np.float32)
if latent.size % 64 != 0:
    raise ValueError(f"latent float count {latent.size} is not divisible by 64 channels: {path}")
num_frames = latent.size // 64
latent = latent.reshape(64, num_frames).T  # [frames, 64 channels]

patch_size = 4
num_patches = num_frames // patch_size

print("=== Intra-patch frame correlation ===")
patch_corrs = []
for p in range(num_patches):
    start = p * patch_size
    end = start + patch_size
    patch = latent[start:end]
    for i in range(patch_size - 1):
        c = np.corrcoef(patch[i], patch[i+1])[0,1]
        patch_corrs.append(c)
    if p < 5:
        cors = [np.corrcoef(patch[i], patch[i+1])[0,1] for i in range(patch_size-1)]
        print(f"  Patch {p}: {cors}")

print(f"\nMean intra-patch corr: {np.mean(patch_corrs):.4f}")

print("\n=== First patch ===")
first_patch = latent[0:4]
for i in range(4):
    print(f"  frame {i}: peak={np.abs(first_patch[i]).max():.4f}, mean={first_patch[i].mean():.4f}")

print(f"\n=== Frame means across all {num_frames} frames ===")
means = [latent[i].mean() for i in range(num_frames)]
print(f"  Range: [{min(means):.4f}, {max(means):.4f}]")
print(f"  Mean: {np.mean(means):.4f}, Std: {np.std(means):.4f}")

# Check if frames are mostly from a narrow distribution
print("\n=== Frame std (measure of variation) ===")
stds = [latent[i].std() for i in range(num_frames)]
print(f"  Range: [{min(stds):.4f}, {max(stds):.4f}]")
print(f"  Mean: {np.mean(stds):.4f}")

# Check correlation between TEXT-ADJACENT frames (where text changes)
# This would catch if the latent actually encodes text content
print("\n=== Latent heatmap statistics ===")
print(f"  Total frames: {latent.shape[0]}")
print(f"  Channels: {latent.shape[1]}")
print(f"  Non-finite values: {np.sum(~np.isfinite(latent))}")
print(f"  Zeros: {np.sum(latent == 0)}")
