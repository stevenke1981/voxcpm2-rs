#!/usr/bin/env python3
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


def inspect_file(path: Path):
    from safetensors import safe_open
    tensors = []
    with safe_open(str(path), framework='pt', device='cpu') as f:
        for name in f.keys():
            t = f.get_tensor(name)
            tensors.append({'name': name, 'shape': list(t.shape), 'dtype': str(t.dtype)})
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256(path),
        'tensor_count': len(tensors),
        'tensors': tensors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    files = []
    for name in ['model.safetensors', 'audiovae.safetensors']:
        p = model_dir / name
        if p.exists():
            files.append(inspect_file(p))
    report = {'model_dir': str(model_dir), 'files': files}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
        print(f'Wrote {args.out}')
    else:
        print(text)


if __name__ == '__main__':
    main()
