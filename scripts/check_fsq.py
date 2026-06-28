import json, sys
sys.path.insert(0, r"D:\VoxCPM\src")
with open(r"E:\voxcpm2_rust_candle_pack\models\VoxCPM2\config.json") as f:
    cfg = json.load(f)

lm = cfg.get("lm_config", {})
print("scalar_quantization_latent_dim:", lm.get("scalar_quantization_latent_dim", "NOT IN CONFIG"))
print("scalar_quantization_scale:", lm.get("scalar_quantization_scale", "NOT IN CONFIG"))

# Load model and check FSQ
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from voxcpm.model.voxcpm2 import VoxCPM2Model
model = VoxCPM2Model.from_local(r"E:\voxcpm2_rust_candle_pack\models\VoxCPM2", device="cpu")
fsq = model.fsq_layer
print(f"\nFSQ in_dim: {fsq.in_dim}, out_dim: {fsq.out_dim}, latent_dim: {fsq.latent_dim}, scale: {fsq.scale}")
print(f"in_proj weight shape: {fsq.in_proj.weight.shape}")
print(f"out_proj weight shape: {fsq.out_proj.weight.shape}")
