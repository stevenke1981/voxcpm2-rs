# AGENTS.md — Controlled Workflow for VoxCPM2 Rust Candle

## Mission

Port VoxCPM2 to Rust/Candle with GPU acceleration and egui GUI. Work must be incremental, testable, and honest about incomplete model layers.

## Rules

1. Do not claim real inference works until milestones C-G pass.
2. Never delete model weights, golden files, or user data.
3. Use small tests first: unit → smoke → manifest → parity → real audio.
4. On shape mismatch, dump tensor name, expected shape, actual shape, source module, and stop.
5. Keep CLI and GUI working after every change.
6. Update docs/todos.md when completing or discovering tasks.
7. For Windows, prefer PowerShell commands in harness.
8. Safety: do not add features for impersonation, fraud, or hiding generated voice identity.

## Gate sequence

```text
G0 cargo fmt/test smoke
G1 inspect assets
G2 tokenizer parity
G3 tensor mapping
G4 submodule shape
G5 audio sanity
G6 quality parity
G7 egui complete flow
G8 GPU benchmark
```

## Allowed automation

- build
- test
- run CLI smoke
- read logs
- create/update source/docs

## Ask/Deny

Ask before:

- downloading multi-GB model files
- installing CUDA/toolchain dependencies
- running long GPU benchmarks

Deny:

- destructive delete commands for model/golden folders
- unsafe pickle loading in Rust
- bypassing safety labels
