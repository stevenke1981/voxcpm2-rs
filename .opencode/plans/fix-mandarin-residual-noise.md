## Plan: fix Mandarin residual noise
**Goal:** Reduce remaining audible artifacts by using the validated Mandarin prompt path, then verify the generated speech with ASR.
**Complexity:** L3

### Sub-tasks
1. [x] Inspect current CLI/GUI defaults and audio polish path -> files: crates/voxcpm2-cli/src/main.rs, crates/voxcpm2-gui/src/main.rs, crates/voxcpm2-core/src/pipeline.rs -> output: root cause confirmed
2. [x] Update Mandarin-safe defaults and warnings -> files: CLI/GUI/core docs -> output: seed 99 and Simplified Chinese prompt guidance
3. [x] Generate GPU speech sample with Simplified Chinese input -> file: output/mandarin_asr_pass_seed99.wav -> output: playable WAV
4. [x] Run audio sanity metrics and ASR transcript check -> output: transcript matches source text
5. [x] Run tests, commit, and push -> output: pushed commit on master

### Risks
| Risk | Mitigation |
|------|------------|
| Traditional Chinese prompt biases Cantonese voice and sounds like noise to Mandarin validation | Use Simplified Chinese source text for Mandarin and warn when Traditional-only characters are detected |
| Seed-dependent CFM noise trajectory causes residual artifacts | Default to seed 99 from the prior controlled seed sweep |
| More low-pass may damage intelligibility | Keep existing conservative audio polish; validate with ASR instead of blindly filtering more |

### Definition of Done
- [x] GPU-generated Mandarin WAV exists and is playable
- [x] ASR transcript matches the intended Simplified Chinese text
- [x] Relevant tests pass
- [x] git commit created and pushed

### Assumptions
- The user wants Mandarin output; therefore test text must be Simplified Chinese, not Traditional Chinese.
