param(
  [string]$ModelDir = "models/VoxCPM2",
  [string]$Device = "cuda",
  [string]$Text = "你好，这是 VoxCPM2 Rust Candle 的 CUDA 速度基准测试。",
  [int]$Steps = 30,
  [int]$Seed = 99,
  [double]$Cfg = 2.5,
  [switch]$Release
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$featureArgs = @("--no-default-features", "--features", "cuda")
$profile = if ($Release) { "--release" } else { "" }

Write-Host "=== VoxCPM2 CUDA RTF Benchmark ===" -ForegroundColor Cyan
Write-Host "Device: $Device | Steps: $Steps | Seed: $Seed"

$exe = if ($Release) { "target\release\voxcpm2.exe" } else { "target\debug\voxcpm2.exe" }
if (-not (Test-Path $exe)) {
  Write-Host "Building..." -ForegroundColor Yellow
  cargo build -p voxcpm2-cli @featureArgs $(if ($Release) { "--release" })
}

$wav = "output/benchmark_$(Get-Date -Format yyyyMMdd_HHmmss).wav"
$metrics = [IO.Path]::ChangeExtension($wav, ".metrics.json")

$sw = [System.Diagnostics.Stopwatch]::StartNew()

& $exe synth `
  --model-dir $ModelDir `
  --device $Device `
  --text $Text `
  --steps $Steps `
  --seed $Seed `
  --cfg $Cfg `
  --out $wav `
  --metrics-out $metrics `
  --i-have-consent  # harmless for synth

$sw.Stop()

$ms = $sw.Elapsed.TotalMilliseconds
Write-Host "Generation time: $([math]::Round($ms/1000,2)) s" -ForegroundColor Green

if (Test-Path $metrics) {
  $m = Get-Content $metrics -Raw | ConvertFrom-Json
  $audioSec = [double]$m.duration_s
  if ($audioSec -gt 0) {
    $rtf = $ms / 1000 / $audioSec
    Write-Host "Audio duration: $([math]::Round($audioSec,2)) s | RTF ≈ $([math]::Round($rtf,3))" -ForegroundColor Green
  }
}

Write-Host "Benchmark done. WAV: $wav" -ForegroundColor Cyan
