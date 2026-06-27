#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_MODEL=0
MODEL_DIR="models/VoxCPM2"
DEVICE="cpu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-model) WITH_MODEL=1; shift ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

cargo fmt --all -- --check
cargo clippy --workspace --features cpu -- -D warnings
cargo test --workspace --features cpu
cargo run -p voxcpm2-cli --features cpu -- synth --text "smoke 測試" --out output/smoke.wav --dry-run

if [[ "$WITH_MODEL" == "1" ]]; then
  cargo run -p voxcpm2-cli --features cpu -- inspect --model-dir "$MODEL_DIR"
  if [[ "$DEVICE" == "cuda" ]]; then
    cargo run -p voxcpm2-cli --no-default-features --features cuda -- benchmark --model-dir "$MODEL_DIR" --device cuda --repeat 1
  elif [[ "$DEVICE" == "metal" ]]; then
    cargo run -p voxcpm2-cli --no-default-features --features metal -- benchmark --model-dir "$MODEL_DIR" --device metal --repeat 1
  fi
fi

echo "Harness completed."
