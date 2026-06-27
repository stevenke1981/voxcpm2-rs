param(
  [ValidateSet("cuda", "metal", "cpu")]
  [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Device -eq "cuda") {
  cargo run -p voxcpm2-cli --no-default-features --features cuda -- synth --text "CUDA smoke" --out output/cuda_smoke.wav --dry-run --device cuda
} elseif ($Device -eq "metal") {
  cargo run -p voxcpm2-cli --no-default-features --features metal -- synth --text "Metal smoke" --out output/metal_smoke.wav --dry-run --device metal
} else {
  cargo run -p voxcpm2-cli --features cpu -- synth --text "CPU smoke" --out output/cpu_smoke.wav --dry-run --device cpu
}
