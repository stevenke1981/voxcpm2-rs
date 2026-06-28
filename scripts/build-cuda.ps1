# Build VoxCPM2 with CUDA support
# Requires: Visual Studio 2019 BuildTools, CUDA 12.x Toolkit
Write-Host "=== Build VoxCPM2 with CUDA ===" -ForegroundColor Cyan

$VS2019_DIR = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
$VCVARS = "$VS2019_DIR\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $VCVARS)) {
    Write-Error "VS 2019 BuildTools not found at: $VCVARS"
    Write-Host "Try: winget install Microsoft.VisualStudio.2019.BuildTools" -ForegroundColor Yellow
    exit 1
}

$cwd = Split-Path -Parent $PSScriptRoot

cmd.exe /c "`"$VCVARS`" && cd /d $cwd && cargo build --release --no-default-features --features cuda 2>&1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Build successful! ===" -ForegroundColor Green
    Write-Host "Run: .\target\release\voxcpm2.exe" -ForegroundColor Cyan
} else {
    Write-Error "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
