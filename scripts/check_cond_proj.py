"""Check cond_proj weight values and bias."""
import json
import numpy as np
import struct

with open('models/VoxCPM2/model.safetensors', 'rb') as f:
    header_len = int.from_bytes(f.read(8), 'little')
    header = json.loads(f.read(header_len).decode('utf-8'))

# Check cond_proj bias
for k in header:
    if 'cond_proj' in k:
        info = header[k]
        print(f"{k}: shape={info['shape']} dtype={info['dtype']}")

def bf16_to_f32_bytes(raw):
    """Convert bfloat16 bytes to float32 numpy array."""
    vals = []
    for i in range(0, len(raw) - 1, 2):
        # bfloat16: upper 16 bits of float32
        u16 = struct.unpack('<H', raw[i:i+2])[0]
        f32 = struct.unpack('<f', struct.pack('<I', u16 << 16))[0]
        vals.append(f32)
    return np.array(vals)

# Read weight
key = 'feat_decoder.estimator.cond_proj.weight'
info = header[key]
offset = info['data_offsets'][0]
end_offset = info['data_offsets'][1]

with open('models/VoxCPM2/model.safetensors', 'rb') as f:
    # Read header
    f.seek(8)  
    header_json = f.read(header_len).decode('utf-8')
    # Add padding from JSON to data start
    data_start = 8 + len(header_json.encode())
    f.seek(data_start + offset)
    raw = f.read(end_offset - offset)

weight = bf16_to_f32_bytes(raw[:20000])
print(f"\ncond_proj.weight stats (first 10000 values of {info['shape']}):")
print(f"  min: {weight.min():.6f}  max: {weight.max():.6f}")
print(f"  mean: {weight.mean():.6f}  std: {weight.std():.6f}")
print(f"  abs_mean: {np.abs(weight).mean():.6f}")
print(f"  fraction near zero (<1e-6): {(np.abs(weight) < 1e-6).mean():.1%}")

# Check if bias exists
if 'feat_decoder.estimator.cond_proj.bias' in header:
    binfo = header['feat_decoder.estimator.cond_proj.bias']
    print(f"\ncond_proj.bias: shape={binfo['shape']} dtype={binfo['dtype']}")
    # Read bias
    boffset = binfo['data_offsets'][0]
    bend = binfo['data_offsets'][1]
    with open('models/VoxCPM2/model.safetensors', 'rb') as f2:
        data_start = 8 + len(json.dumps(header).encode())
        f2.seek(data_start + boffset)
        bias_raw = f2.read(bend - boffset)
    bias = bf16_to_f32_bytes(bias_raw)
    print(f"  bias min: {bias.min():.6f}  max: {bias.max():.6f}")
    print(f"  bias mean: {bias.mean():.6f}  std: {bias.std():.6f}")
else:
    print("\nNo cond_proj bias found.")

# Compare with feat_encoder.in_proj for reference
print("\n--- feat_encoder.in_proj.weight (also [1024, 64]) ---")
for k in header:
    if 'feat_encoder.in_proj.weight' in k:
        info2 = header[k]
        offset2 = info2['data_offsets'][0]
        end2 = info2['data_offsets'][1]
        with open('models/VoxCPM2/model.safetensors', 'rb') as f3:
            data_start = 8 + len(json.dumps(header).encode())
            f3.seek(data_start + offset2)
            raw2 = f3.read(end2 - offset2)
        w2 = bf16_to_f32_bytes(raw2[:20000])
        print(f"  shape={info2['shape']}")
        print(f"  min: {w2.min():.6f}  max: {w2.max():.6f}")
        print(f"  mean: {w2.mean():.6f}  std: {w2.std():.6f}")
        print(f"  abs_mean: {np.abs(w2).mean():.6f}")
        break
