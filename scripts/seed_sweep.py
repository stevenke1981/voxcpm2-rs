"""Seed sweep: test seeds 0-99 with cfg=2.5 and measure speech energy."""
import subprocess
import sys
import os
import tempfile

# Ensure output dir
os.makedirs('output/seed_sweep', exist_ok=True)

try:
    import soundfile as sf
    import numpy as np
    from scipy import signal
    HAVE_AUDIO = True
except ImportError:
    HAVE_AUDIO = False
    print("WARNING: soundfile/scipy not available, using file-size heuristic")

results = []
for seed in range(100):
    out = f'output/seed_sweep/s{seed}.wav'
    
    # Run inference
    cmd = [
        'cargo', 'run', '--release', '--features', 'cuda',
        '--bin', 'voxcpm2', '--',
        'synth',
        '--text', '这是修正后的语音现在应该更清楚',
        '--out', out,
        '--seed', str(seed),
        '--cfg', '2.5',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  seed={seed}: FAILED (rc={result.returncode})')
        continue
    
    # Check output file
    if not os.path.exists(out) or os.path.getsize(out) < 1000:
        print(f'  seed={seed}: no valid output')
        continue
    
    if HAVE_AUDIO:
        try:
            data, sr = sf.read(out)
            f, t, Zxx = signal.stft(data, sr, nperseg=2048)
            mag = np.abs(Zxx)
            total = mag.sum()
            if total > 0:
                lo = mag[(f >= 100) & (f < 300)].sum() / total * 100
                mid = mag[(f >= 300) & (f < 8000)].sum() / total * 100
                hi = mag[(f >= 8000) & (f < sr//2)].sum() / total * 100
                peak = np.abs(data).max()
                print(f'  seed={seed:3d}: speech={mid:.1f}% rumble={lo:.1f}% hf={hi:.1f}% peak={peak:.4f}')
                results.append((seed, mid, lo, hi, peak))
            else:
                print(f'  seed={seed:3d}: silent output')
        except Exception as e:
            print(f'  seed={seed:3d}: error: {e}')
    else:
        size_kb = os.path.getsize(out) / 1024
        print(f'  seed={seed:3d}: size={size_kb:.0f} KB')
        results.append((seed, size_kb))

# Summary
if HAVE_AUDIO and results:
    results.sort(key=lambda x: -x[1])  # Sort by speech energy descending
    print('\n' + '=' * 70)
    print('TOP 10 SEEDS BY SPEECH ENERGY')
    print('=' * 70)
    for i, (seed, mid, lo, hi, peak) in enumerate(results[:10]):
        print(f'  #{i+1}: seed={seed:3d} speech={mid:.1f}% rumble={lo:.1f}% hf={hi:.1f}% peak={peak:.4f}')
    print(f'\nTotal valid outputs: {len(results)}/{100}')
    
    # Stats
    speech_values = [r[1] for r in results]
    print(f'Speech energy: mean={np.mean(speech_values):.1f}% std={np.std(speech_values):.1f}% '
          f'min={min(speech_values):.1f}% max={max(speech_values):.1f}%')
