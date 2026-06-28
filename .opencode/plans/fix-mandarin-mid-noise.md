## Plan: fix Mandarin mid-file noise
**Goal:** Remove the audible middle-section noise in `output/mandarin_bg_gate_seed99.wav` without hurting Mandarin intelligibility.
**Complexity:** L3

### Sub-tasks
1. [x] Compare same-seed old/new Mandarin WAVs at frame and band level -> output: exact noisy time range and root cause
2. [x] Adjust audio polish narrowly -> files: crates/voxcpm2-core/src/audio.rs, crates/voxcpm2-core/src/pipeline.rs if needed -> output: less mid-section artifact
3. [x] Regenerate Mandarin sample with same text/seed -> output: replacement WAV for listening
4. [x] Verify WAV sanity, metrics, ASR, and Rust tests -> output: evidence that noise is reduced and text remains recognizable
5. [x] Update docs/lessons, commit, and push -> output: pushed commit on master

### Risks
| Risk | Mitigation |
|------|------------|
| A hard gate can create pumping or musical-noise artifacts | Prefer smoother release/lookahead or targeted de-click/silence handling |
| Removing mid noise can damage consonants | Verify with ASR and compare speech-band metrics |
| Different seed/stop timing hides the real cause | Keep text, seed, cfg, steps, and max-ar-steps fixed |

### Definition of Done
- [x] Middle-section noise cause identified
- [x] Improved sample generated
- [x] ASR still recognizes the intended Mandarin text
- [x] Tests pass
- [x] git commit created and pushed

### Assumptions
- `mandarin_asr_pass_seed99.wav` and `mandarin_bg_gate_seed99.wav` are comparable same-seed samples with only polish changes.
