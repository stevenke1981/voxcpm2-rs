param(
  [switch]$WithModel,
  [string]$ModelDir = "models/VoxCPM2",
  [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Run($cmd) {
  Write-Host "`n>>> $cmd" -ForegroundColor Cyan
  Invoke-Expression $cmd
}

Run "cargo fmt --all -- --check"
Run "cargo clippy --workspace --features cpu -- -D warnings"
Run "cargo test --workspace --features cpu"
Run "cargo run -p voxcpm2-cli --features cpu -- synth --text 'smoke 測試' --out output/smoke.wav --dry-run"

if ($WithModel) {
  Run "cargo run -p voxcpm2-cli --features cpu -- inspect --model-dir `"$ModelDir`""
  if ($Device -eq "cuda") {
    Run "cargo run -p voxcpm2-cli --no-default-features --features cuda -- benchmark --model-dir `"$ModelDir`" --device cuda --repeat 1"
  } elseif ($Device -eq "metal") {
    Run "cargo run -p voxcpm2-cli --no-default-features --features metal -- benchmark --model-dir `"$ModelDir`" --device metal --repeat 1"
  }
}

Write-Host "`nHarness completed." -ForegroundColor Green
