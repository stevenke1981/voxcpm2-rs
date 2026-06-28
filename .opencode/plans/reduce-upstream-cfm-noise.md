## Plan: reduce upstream CFM noise
**Goal:** Identify and reduce the remaining audible noise after output polishing by fixing upstream CFM/latent behavior where possible.
**Complexity:** L3

### Sub-tasks
1. [x] Measure a fresh CUDA sample with the current dirty worktree -> output: current noise metrics and ASR.
2. [x] Compare Rust CFM scheduler/CFG/noise path against official Python -> files: `unified_cfm.rs`, upstream `unified_cfm.py` -> output: root cause.
3. [x] Implement the smallest upstream improvement that reduces residual noise without breaking intelligibility.
4. [x] Verify with unit tests, CUDA synth, WAV metrics, and ASR.
5. [x] Update todos/lessons with the root cause and validation evidence.
6. [x] git commit created and pushed.

### Risks
| Risk | Mitigation |
|------|------------|
| Existing uncommitted fixes are user work | Preserve and build on them; stage only relevant files. |
| Noise is stochastic and varies per run | Use comparable spectral metrics plus ASR, not one subjective listen. |
| Aggressive filtering can muffle speech | Prefer CFM/default parameter fixes before heavier output filtering. |

### Definition of Done
- [x] A specific upstream or output-stage cause is documented.
- [x] New sample has lower audible-noise metric or safer peak/ultra-high-frequency behavior.
- [x] Speech remains ASR-intelligible.
- [x] Tests pass.
- [x] Changes are committed and pushed.

### Assumptions
- The current issue is residual audible noise after intelligible speech, not total failure to speak.
