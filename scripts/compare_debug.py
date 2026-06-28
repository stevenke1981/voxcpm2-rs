"""Compare Rust vs Python per-step debug tensors."""
import numpy as np
import os
import struct

RUST_DIR = 'output'
PY_DIR = 'output/py_ref'

def load_f32(path):
    return np.fromfile(path, dtype=np.float32)

def compare_tensor(name, py_path, rust_path):
    if not os.path.exists(py_path):
        print(f'  [SKIP] Python missing: {py_path}')
        return
    if not os.path.exists(rust_path):
        print(f'  [SKIP] Rust missing: {rust_path}')
        return
    py_t = load_f32(py_path)
    rust_t = load_f32(rust_path)
    
    if py_t.shape != rust_t.shape:
        print(f'  [SHAPE MISMATCH] {name}: py={py_t.shape} rust={rust_t.shape}')
        # Truncate to min length
        min_len = min(len(py_t), len(rust_t))
        py_t = py_t[:min_len]
        rust_t = rust_t[:min_len]
    
    # Stats
    diff = py_t - rust_t
    abs_diff = np.abs(diff)
    max_diff = abs_diff.max()
    mean_diff = diff.mean()
    std_diff = diff.std()
    
    # Cosine similarity
    py_norm = np.linalg.norm(py_t)
    rust_norm = np.linalg.norm(rust_t)
    if py_norm > 0 and rust_norm > 0:
        cos_sim = np.dot(py_t, rust_t) / (py_norm * rust_norm)
    else:
        cos_sim = 0.0
    
    print(f'  [COMPARE] {name}')
    print(f'    py:  mean={py_t.mean():.6f} std={py_t.std():.6f} peak={np.abs(py_t).max():.6f}')
    print(f'    rust:mean={rust_t.mean():.6f} std={rust_t.std():.6f} peak={np.abs(rust_t).max():.6f}')
    print(f'    diff:max_abs={max_diff:.6f} mean={mean_diff:.6f} std={std_diff:.6f}')
    print(f'    cos_sim={cos_sim:.6f}')
    
    # MAE / RMSE
    mae = abs_diff.mean()
    rmse = np.sqrt((diff ** 2).mean())
    print(f'    MAE={mae:.6f} RMSE={rmse:.6f}')
    
    # Find top-3 largest absolute differences
    if len(abs_diff) > 0:
        top_indices = np.argsort(abs_diff)[-5:][::-1]
        for idx in top_indices:
            print(f'    top diff at [{idx}]: py={py_t[idx]:.6f} rust={rust_t[idx]:.6f} diff={diff[idx]:.6f}')
    
    return max_diff, cos_sim

print('=' * 70)
print('PREFILL STATE COMPARISON')
print('=' * 70)
compare_tensor('tslm_init', f'{PY_DIR}/debug_tslm_init.f32', f'{RUST_DIR}/debug_tslm_init.f32')
compare_tensor('ralm_init', f'{PY_DIR}/debug_ralm_init.f32', f'{RUST_DIR}/debug_ralm_init.f32')

print()
print('=' * 70)
print('PER-STEP COMPARISON (first 18 steps - Python)')
print('=' * 70)

tensors = ['hlm_before', 'hres_before', 'mu_input', 'pred_feat', 'currembed']

for step in range(18):
    print(f'\n--- Step {step} ---')
    for tname in tensors:
        py_t = f'{PY_DIR}/debug_step{step}_{tname}.f32'
        rust_t = f'{RUST_DIR}/debug_step{step}_{tname}.f32'
        if os.path.exists(py_t) and os.path.exists(rust_t):
            max_diff, cos_sim = compare_tensor(tname, py_t, rust_t)
        elif os.path.exists(py_t):
            print(f'  [SKIP] {tname}: Python exists, Rust missing')
        else:
            print(f'  [SKIP] {tname}: Python missing')

print()
print('=' * 70)
print('FINAL LATENT COMPARISON')
print('=' * 70)
compare_tensor('latent', f'{PY_DIR}/latent_py.f32', f'{RUST_DIR}/latent_rust.f32')
