#!/usr/bin/env python3
"""Convert VoxCPM2 audiovae.pth to audiovae.safetensors.

This script intentionally uses Python/PyTorch because .pth is pickle-based.
Do not load untrusted .pth files.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def flatten_state(obj):
    if isinstance(obj, dict):
        for key in ('state_dict', 'model', 'module'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise TypeError(f'Unsupported pth root type: {type(obj)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--manifest', default=None)
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading {src} ...')
    obj = torch.load(src, map_location='cpu')
    state = flatten_state(obj)
    tensors = {}
    skipped = []
    for k, v in state.items():
        if torch.is_tensor(v):
            tensors[k] = v.detach().cpu().contiguous()
        else:
            skipped.append(k)
    if not tensors:
        raise SystemExit('No tensors found in pth file')
    save_file(tensors, str(dst))

    manifest = {
        'input': str(src),
        'output': str(dst),
        'input_sha256': sha256(src),
        'output_sha256': sha256(dst),
        'tensor_count': len(tensors),
        'skipped_keys': skipped,
        'tensors': [
            {'name': k, 'shape': list(v.shape), 'dtype': str(v.dtype)}
            for k, v in tensors.items()
        ],
    }
    manifest_path = Path(args.manifest) if args.manifest else dst.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {dst}')
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
