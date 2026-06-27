$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
cargo run -p voxcpm2-cli --features cpu -- synth --text "VoxCPM2 Rust Candle smoke" --out output/smoke_cli.wav --dry-run
Get-Item output/smoke_cli.wav | Format-List Name,Length,LastWriteTime
