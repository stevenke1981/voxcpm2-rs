#!/usr/bin/env python3
"""Download VoxCPM2 model files from Hugging Face.

Usage:
  python scripts/download_model.py --repo openbmb/VoxCPM2 --out models/VoxCPM2
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='openbmb/VoxCPM2')
    ap.add_argument('--out', default='models/VoxCPM2')
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit('Please install: pip install huggingface_hub') from e

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        allow_patterns=[
            'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'tokenization_voxcpm2.py',
            'model.safetensors', 'audiovae.pth', 'README.md'
        ],
    )
    print(f'Downloaded {args.repo} to {out}')


if __name__ == '__main__':
    main()
