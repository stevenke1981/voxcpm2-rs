@echo off
REM Build VoxCPM2 with CUDA support
REM Requires: Visual Studio 2019 BuildTools, CUDA 12.x Toolkit
echo === Build VoxCPM2 with CUDA ===

set "VS2019_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
set "VCVARS=%VS2019_DIR%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS%" (
    echo Error: VS 2019 BuildTools not found at %VCVARS%
    echo Try: winget install Microsoft.VisualStudio.2019.BuildTools
    exit /b 1
)

call "%VCVARS%"
cd /d "%~dp0.."
cargo build --release --no-default-features --features cuda
if %ERRORLEVEL% EQU 0 (
    echo === Build successful! ===
    echo Run: .\target\release\voxcpm2.exe
) else (
    echo Build failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
