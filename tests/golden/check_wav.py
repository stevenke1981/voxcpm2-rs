#!/usr/bin/env python3
import argparse
import wave
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('wav')
ap.add_argument('--sample-rate', type=int, default=48000)
args = ap.parse_args()
path = Path(args.wav)
with wave.open(str(path), 'rb') as w:
    assert w.getframerate() == args.sample_rate, (w.getframerate(), args.sample_rate)
    assert w.getnchannels() == 1
    assert w.getnframes() > 0
print(f'OK: {path} {args.sample_rate}Hz frames={w.getnframes()}')
