## Plan: fix synth and clone background noise
**Goal:** Reduce audible background noise for both text synthesis and voice clone outputs with measurable audio gates.
**Complexity:** L3

### Sub-tasks
1. [x] Inspect current audio polish and clone reference path -> files: crates/voxcpm2-core/src/audio.rs, crates/voxcpm2-core/src/pipeline.rs -> output: root cause hypothesis
2. [x] Measure current synth and clone outputs when available -> output: background/quiet-frame metrics
3. [x] Implement a shared conservative cleanup that applies to synth and clone outputs -> files: audio.rs/pipeline.rs -> output: lower quiet/background noise without hurting speech
4. [x] Verify with unit tests, WAV sanity, and ASR when generated samples are available -> output: evidence that speech remains intelligible
5. [x] Update docs/lessons, commit, and push -> output: pushed commit on master

### Risks
| Risk | Mitigation |
|------|------------|
| Over-aggressive noise gating clips consonants or tail phonemes | Use a slow adaptive floor, preserve loud speech, and validate with ASR |
| Clone noise comes from reference audio, not only final output | Pre-clean reference audio before AudioVAE encoding and still run final shared polish |
| Existing generated samples vary by seed or stop timing | Use bounded max steps and compare metrics on same sample family |

### Definition of Done
- [x] Shared synth/clone output cleanup is implemented
- [x] Clone reference audio is pre-cleaned before encoding
- [x] Relevant tests pass
- [x] git commit created and pushed

### Assumptions
- Background noise is low-level non-speech energy in quiet frames and/or reference-audio noise copied into clone conditioning.
