<#
.SYNOPSIS
  Build Windows portable zip for voxcpm2-rust-candle (CPU by default for widest compatibility).
  Includes the CLI binary, basic docs, example config, and safety/FAQ stubs.
  User must download models/ separately (see README).
#>
param(
  [string]$OutDir = "dist",
  [switch]$IncludeCudaNote
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$version = (Get-Date -Format "yyyyMMdd")
$pkgName = "voxcpm2-rust-candle-portable-$version"
$pkgDir = Join-Path $OutDir $pkgName
New-Item -ItemType Directory -Force -Path $pkgDir | Out-Null

Write-Host "Building release (cpu) for portable..." -ForegroundColor Cyan
cargo build -p voxcpm2-cli --release --features cpu

$binSrc = "target\release\voxcpm2.exe"
if (-not (Test-Path $binSrc)) { throw "Release binary not found at $binSrc" }

Copy-Item $binSrc -Destination (Join-Path $pkgDir "voxcpm2.exe") -Force

# Bundle minimal docs
Copy-Item README.md -Destination (Join-Path $pkgDir "README.txt") -Force
if (Test-Path configs/voxcpm2.local.example.toml) {
  Copy-Item configs/voxcpm2.local.example.toml -Destination (Join-Path $pkgDir "config.example.toml") -Force
}

# Safety and quick start
@"
VoxCPM2 Rust/Candle Portable (Windows)

SAFETY NOTICE
- Voice cloning is for authorized use only. You must have explicit rights/consent for any reference voice.
- Do NOT use for impersonation, fraud, disinformation, or any illegal purpose.
- All outputs are labeled by default in the app/CLI. Clearly mark AI-generated audio in any public use.
- This software is provided under Apache-2.0. See full license in source.

QUICK START (CPU)
1. Download models from https://huggingface.co/openbmb/VoxCPM2 (or ModelScope) into models/VoxCPM2/
2. Convert AudioVAE if needed: python scripts/convert_audiovae_pth_to_safetensors.py --input models/VoxCPM2/audiovae.pth --output models/VoxCPM2/audiovae.safetensors
3. Run: voxcpm2.exe synth --text "你好，这是测试。" --out demo.wav --seed 99 --cfg 2.5 --steps 30

For CUDA build: cargo build -p voxcpm2-cli --release --no-default-features --features cuda (requires VS Build Tools + CUDA)

See full README for GUI, clone (with --i-have-consent), and safety details.
"@ | Set-Content -LiteralPath (Join-Path $pkgDir "SAFETY_AND_QUICKSTART.txt") -Encoding UTF8

# Copy a minimal FAQ excerpt
@"
FAQ
- Slow on CPU? Yes, expect minutes for full quality. Use CUDA build for speed.
- Noise in output? Use --seed 99 --cfg 2.5 --steps 30 and ensure models are correctly converted.
- Clone requires consent flag and legal right to the voice.
- Models are ~4.6 GB (not bundled).
"@ | Set-Content -LiteralPath (Join-Path $pkgDir "FAQ.txt") -Encoding UTF8

# Zip it
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$zipPath = Join-Path $OutDir "$pkgName.zip"
Compress-Archive -Path $pkgDir -DestinationPath $zipPath -Force

Write-Host "Portable package created: $zipPath" -ForegroundColor Green
Write-Host "Size: $([math]::Round((Get-Item $zipPath).Length / 1MB, 1)) MB (binary only; models separate)" -ForegroundColor Yellow
