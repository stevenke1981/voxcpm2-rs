"""Generate ConvTranspose1d test data for Candle verification."""
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(42)
ckpt = torch.load('models/VoxCPM2/audiovae.pth', map_location='cpu', weights_only=True)
sd = ckpt['state_dict']


def save_test(block_name, in_ch, out_ch, k, s):
    g = sd[f'decoder.{block_name}.weight_g']
    v = sd[f'decoder.{block_name}.weight_v']
    vn = torch.sqrt((v**2).sum(dim=(1, 2), keepdim=True))
    w = (g * v / vn).numpy().astype(np.float32)
    b = sd[f'decoder.{block_name}.bias'].numpy().astype(np.float32)

    if k > s:
        pad = (k - s + 1) // 2  # ceil division
    else:
        pad = 0
    op = s - k + 2 * pad
    if op < 0:
        op = 0

    L_in = 16
    L_out_calc = (L_in - 1) * s + k - 2 * pad + op
    if L_out_calc != s * L_in:
        print(f'WARNING: {block_name}: L_in={L_in}, expected {s*L_in}, got {L_out_calc}')

    x = torch.randn(1, in_ch, 16)
    w_pt = torch.tensor(w)
    b_pt = torch.tensor(b)
    out = F.conv_transpose1d(x, w_pt, bias=b_pt, stride=s, padding=pad, output_padding=op)

    safe_name = block_name.replace('.', '_')
    np.save(f'data/{safe_name}_input.npy', x.numpy().astype(np.float32))
    np.save(f'data/{safe_name}_weight.npy', w)
    np.save(f'data/{safe_name}_bias.npy', b)
    np.save(f'data/{safe_name}_expected.npy', out.numpy().astype(np.float32))
    np.save(f'data/{safe_name}_x.npy', x.numpy().astype(np.float32))
    np.save(f'data/{safe_name}_w.npy', w)
    np.save(f'data/{safe_name}_b.npy', b)
    np.save(f'data/{safe_name}_y.npy', out.numpy().astype(np.float32))

    print(f'{block_name}: ok, shape={list(out.shape)}, peak={out.abs().max().item():.6f},'
          f' pad={pad}, op={op}, in_ch={in_ch}, out_ch={out_ch}, k={k}, s={s}')


import os
os.makedirs('data', exist_ok=True)

for item in [
    ('model.5.block.1', 256, 128, 4, 2),
    ('model.6.block.1', 128, 64, 4, 2),
    ('model.7.block.1', 64, 32, 4, 2),
    ('model.2.block.1', 2048, 1024, 16, 8),
    ('model.3.block.1', 1024, 512, 12, 6),
    ('model.4.block.1', 512, 256, 10, 5),
]:
    save_test(*item)
