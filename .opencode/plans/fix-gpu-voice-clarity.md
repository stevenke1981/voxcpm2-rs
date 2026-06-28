## Plan: fix gpu voice clarity
**Goal:** Improve unclear GPU speech by aligning Rust autoregressive conditioning with the official VoxCPM2 Python inference loop.
**Complexity:** L3

### Sub-tasks
1. [x] Compare Rust autoregressive flow with `D:\VoxCPM\src\voxcpm\model\voxcpm2.py` `_inference()` -> file: `crates/voxcpm2-core/src/autoregressive.rs` -> output: root-cause notes in code/docs.
2. [x] Fix residual LM initial input and per-step FSQ state -> file: `crates/voxcpm2-core/src/autoregressive.rs` -> output: DiT conditioning matches Python order.
3. [x] Add focused test for stop-gate minimum step behavior -> file: `crates/voxcpm2-core/src/autoregressive.rs` -> output: off-by-one is locked.
4. [x] Update project task notes -> file: `docs/todos.md` or `todos.md` -> output: diagnosis and completed fix recorded.
5. [x] Verify with fmt/tests and a CUDA synth smoke -> output: real WAV and command results.

### Risks
| Risk | Mitigation |
|------|------------|
| Full model GPU synth is slow | Use short text, low steps, bounded `--max-ar-steps`, then run normal unit gates. |
| Existing unrelated local changes | Do not stage unrelated `.opencode/status-footer/state.json` or `.codebase-memory/` changes. |
| Audio quality cannot be fully judged by stats | Report objective latent/audio stats and save a fresh WAV for listening. |

### Definition of Done
- [x] Rust autoregressive loop matches Python residual/FSQ update order for the fixed sections.
- [x] Focused tests pass.
- [x] Modified Rust file passes `rustfmt --check`.
- [x] Real CUDA synth produces a WAV without runtime errors.
- [ ] git commit created and pushed.

### Assumptions
- Local model files under `models/VoxCPM2` are complete.
- The installed CUDA/MSVC environment remains usable through `run_with_vs.cmd` or existing Cargo config.
