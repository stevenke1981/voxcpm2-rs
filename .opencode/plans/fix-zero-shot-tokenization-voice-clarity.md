## Plan: fix zero-shot tokenization voice clarity
**Goal:** Make Rust zero-shot synthesis use the same text token sequence as official VoxCPM2 so generated speech matches the requested text.
**Complexity:** L3

### Sub-tasks
1. [x] Reproduce unclear speech with ASR -> output: Rust WAV transcribes to unrelated/repeated text.
2. [x] Compare official zero-shot token path with Rust pipeline -> file: `crates/voxcpm2-core/src/pipeline.rs` -> output: root cause.
3. [x] Replace chat-template prompt with `target_text + <|audio_start|>` zero-shot tokens -> file: `crates/voxcpm2-core/src/pipeline.rs` -> output: Python-matching text/audio masks for no-reference mode.
4. [x] Add focused tokenizer/pipeline test -> output: zero-shot tokens append audio_start and do not add chat/BOS.
5. [x] Verify with CUDA synth + ASR, then tests.

### Risks
| Risk | Mitigation |
|------|------------|
| ASR runner may fail at combine stage | Read per-chunk raw transcript and use it as the intelligibility signal. |
| Voice design control changes token text | Match official CLI: `({control}){text}` before tokenization. |
| Full ASR/GPU synth is slow | Use one bounded 30-step sample for validation, then keep unit tests targeted. |

### Definition of Done
- [x] Rust zero-shot token sequence matches official Python mode structurally.
- [x] New WAV ASR is closer to target text than previous unrelated transcript.
- [x] Focused tests pass.
- [x] CUDA synth succeeds.
- [x] git commit created and pushed.

### Assumptions
- Current task is no-reference zero-shot/design synthesis, not cloning/continuation mode.
