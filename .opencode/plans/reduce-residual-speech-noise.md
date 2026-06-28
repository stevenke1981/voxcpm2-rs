## Plan: reduce residual speech noise
**Goal:** Identify and reduce the remaining low-level noise in generated VoxCPM2 speech while preserving intelligible voice.
**Complexity:** L3

### Sub-tasks
1. [x] Measure current generated WAV noise and spectrum -> output: baseline metrics.
2. [x] Inspect audio output path and upstream latent stats -> files: `crates/voxcpm2-core/src/audio.rs`, `crates/voxcpm2-core/src/pipeline.rs` -> output: likely root cause.
3. [x] Implement a conservative audio post-filter/default that targets residual hiss without masking speech -> output: cleaner WAV.
4. [x] Verify with CUDA synth, WAV sanity, ASR, and focused tests.
5. [x] Update docs/todos and lessons if a durable rule is learned.
6. [x] git commit created and pushed.

### Risks
| Risk | Mitigation |
|------|------------|
| Denoising can muffle consonants | Keep filter conservative and validate ASR transcript after the change. |
| Noise may be latent/runtime, not output PCM | Measure both latent/audio stats and avoid claiming model parity is solved. |
| GPU synth is slow | Use one bounded validation sample and smaller unit tests for code behavior. |

### Definition of Done
- [x] Root cause is documented in repo notes.
- [x] Generated sample has lower residual noise metric or less high-frequency hiss.
- [x] Speech remains intelligible by ASR/manual expectation.
- [x] Relevant tests pass.
- [x] Changes are committed and pushed.

### Assumptions
- Current issue is residual hiss/noise after intelligible speech, not the previous wrong-text failure.
