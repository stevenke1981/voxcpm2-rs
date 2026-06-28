"""Verify Rust AudioVAE decoder by decoding the saved latent with Python AudioVAE."""
import sys
sys.path.insert(0, r"D:\VoxCPM\src")

import numpy as np
import torch
import safetensors.torch
from pathlib import Path

def main():
    model_dir = Path(r"E:\voxcpm2_rust_candle_pack\models\VoxCPM2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load AudioVAE with config
    from voxcpm.modules.audiovae.audio_vae_v2 import AudioVAE, AudioVAEConfig
    config = AudioVAEConfig()
    vae = AudioVAE(config)

    # Load ALL weights from safetensors (includes encoder + decoder)
    sd = safetensors.torch.load_file(str(model_dir / "audiovae.safetensors"))
    
    # Apply with strict=False (we only care about decoder, encoder may have partial keys)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"  Loaded AudioVAE: {len(sd)} keys, {len(missing)} missing (expected: encoder), {len(unexpected)} unexpected")
    vae = vae.to(device)
    vae.eval()

    # Load Rust latent
    latent = np.fromfile(str(model_dir.parent.parent / "output" / "latent_rust.f32"), dtype=np.float32)
    print(f"  Loaded {len(latent)} floats")
    
    latent = torch.from_numpy(latent).float()
    B, C, T = 1, 64, latent.numel() // 64
    latent = latent.reshape(C, T).unsqueeze(0).to(device)
    print(f"  Latent shape: {list(latent.shape)}, peak={latent.abs().max().item():.6f}")

    # Decode with Python AudioVAE at 48000 Hz
    with torch.no_grad():
        output = vae.decode(latent)  # Returns tensor [B, 1, samples]
        print(f"  Output type: {type(output)}, shape: {list(output.shape)}")
        wav = output.cpu().numpy().flatten()
    
    print(f"  Python decoded: {len(wav)} samples, peak={np.abs(wav).max():.6f}")
    print(f"  Stats: mean={wav.mean():.6f}, std={wav.std():.6f}")

    # Save waveforms
    wav.astype(np.float32).tofile(str(model_dir.parent.parent / "output" / "python_decoded.f32"))
    import soundfile as sf
    sf.write(str(model_dir.parent.parent / "output" / "python_decoded.wav"), wav, 48000)
    print("  Saved python_decoded.wav")

if __name__ == "__main__":
    main()
