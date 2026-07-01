# Knowledge Graph

## 2026-07-01 - VoxCPM clone generation alignment

### Sources

- OpenBMB/VoxCPM `VoxCPM/src/voxcpm/model/voxcpm2.py` at `b9fbaecf87c4eabfd7c424b745152d9d2b778f5b`
- stevenke1981/voxcpm-cpp `src/voxcpm.c` and `src/sequence.c` at `896f59596158911c57ab33826b1169f450f89e78`

### Flow Nodes

- `SynthRequest`
  - Owns text, generation controls, optional `ref_audio_path`, optional `prompt_audio_path`, and optional `prompt_text`.
  - `has_audio_conditioning()` is true when either reference or prompt audio exists.
- `voxcpm2-cli clone`
  - Accepts reference-only, prompt-only continuation, and reference+prompt combined modes.
  - Requires `--i-have-consent`.
  - Requires `--prompt-text` when `--prompt-audio` is present.
- `VoxPipeline::synthesize_real`
  - Tokenizes `prompt_text + target_text` when prompt continuation is active.
  - Builds clone prefix before autoregressive generation.
- `encode_clone_prefix`
  - Reference branch: right-pads audio, emits `ref_audio_start`, reference audio patch embeddings, `ref_audio_end`.
  - Text branch: emits tokenized `prompt_text + target_text + audio_start`.
  - Prompt branch: left-pads audio and appends prompt patch embeddings after text/audio start.
  - Returns prompt last latent patch as `initial_prev_feat`.
- `generate_autoregressive_clone`
  - Accepts `initial_prev_feat`.
  - Uses it as the first CFM previous-feature condition for prompt continuation.

### Parity Edges

- `OpenBMB _encode_wav(padding_mode="right")` -> `encode_clone_audio_conditioning(..., CloneAudioPadding::Right)`
- `OpenBMB _encode_wav(padding_mode="left")` -> `encode_clone_audio_conditioning(..., CloneAudioPadding::Left)`
- `OpenBMB text = prompt_text + target_text` -> `build_prompt_target_text(prompt_text, target_text)`
- `OpenBMB prefix_feat_cond = feat[:, -1, ...]` -> `ClonePrefixResult.initial_prev_feat`
- `voxcpm-cpp vcpm_seq_build_clone` -> `encode_clone_prefix` sequence order

### Remaining Edges

- `GUI CloneTab` currently maps only reference-only mode; prompt-only and combined controls are not exposed yet.
- Real CUDA + ASR gate must still validate prompt-only and combined output quality against accepted baseline.

## 2026-07-01 - Target-length cap and placeholder parity

### Sources

- OpenBMB/VoxCPM `src/voxcpm/model/voxcpm2.py` at `07c937b2956c5c122d736cfa1a9cd769965bfe5a`
- stevenke1981/voxcpm-cpp `src/sequence.c` at `896f59596158911c57ab33826b1169f450f89e78`

### Updated Edges

- `OpenBMB target_text_token length` -> `generate_autoregressive(... max_len_base_tokens)`
- `OpenBMB max_len=min(target_text_length*6+10,max_len)` -> `generate_autoregressive_with max_len_base`
- `OpenBMB/C++ audio patch placeholder 0` -> `encode_clone_prefix audio_patch_token_id=0`

### Evidence

- `output/alignment_synth_short_seed102.wav` passed CUDA ASR for the main text.
- `output/alignment_combined_seed102_v2.wav` logged `seq_len=273 max_len_base=30 max_len=190`.
- Combined clone still needs follow-up because stop head did not fire before the target-length cap.

## 2026-07-01 - Patch-boundary de-switch audio polish

### Symptom

- User-reported generated samples sounded like FM radio tuning or station switching.
- Same seed/text analysis showed the largest discontinuities at ~160ms multiples, matching
  `patch_size=4` latent frames decoded by AudioVAE at 48kHz output.

### Flow Nodes

- `polish_generated_speech`
  - Runs after AudioVAE decode and before edge fades/headroom limiting.
  - Now invokes `apply_patch_boundary_smoother`.
- `apply_patch_boundary_smoother`
  - Scans 160ms patch boundaries.
  - Detects boundary-local high-frequency RMS ratio, full-band RMS ratio, ZCR delta, and sample-step jumps.
  - Only smooths triggered boundaries by lowering high-frequency residual around the boundary.
- `AudioPolishReport.patch_boundaries_smoothed`
  - Exposes how many boundaries were touched in CLI metrics and stderr logs.

### Evidence

- `output/alignment_synth_short_seed102_patchsmooth.wav` generated on CUDA with seed 102.
- Metrics: `patch_boundaries_smoothed=9`, `quiet_rms_before=0.01100325`,
  `quiet_rms_after=0.0015115119`, `peak_after=0.5491458`.
- Boundary comparison vs `output/alignment_synth_short_seed102.wav`:
  max sample-step `0.0631 -> 0.0343`; worst high-frequency ratio `7.22 -> 5.62`.
- faster-whisper large-v3-turbo CUDA ASR preserved the transcript:
  `这是普通话测试` / `声音清楚自然`.

## 2026-07-01 - High-band residual background leveler

### Symptom

- `output/alignment_synth_short_seed102_patchsmooth.wav` still had background switching/tuning noise.
- A 20ms analysis window showed high-band residual jumps independent of simple sample clicks, with top
  post-frame high-band RMS up to `0.03085`. The window size is a diagnostic choice from this analysis.

### Flow Nodes

- `polish_generated_speech`
  - Runs `apply_highband_residual_leveler` after patch-boundary smoothing.
- `apply_highband_residual_leveler`
  - Splits speech at 4kHz into low/mid voice body and high-band residual.
  - Computes per-20ms high-band target from full-band RMS, with a fixed maximum cap.
  - Uses fast attack and slower release to stop background noise from opening abruptly.
- `AudioPolishReport.highband_frames_leveled`
  - Reports how many frames had high-band residual attenuation.

### Evidence

- `output/alignment_synth_short_seed102_highband.wav` generated on CUDA with seed 102.
- Metrics: `patch_boundaries_smoothed=9`, `highband_frames_leveled=54`,
  `peak_after=0.5435606`, `quiet_rms_after=0.0015115119`.
- Compared with `alignment_synth_short_seed102_patchsmooth.wav`:
  top 20ms post high-band RMS max `0.03085 -> 0.01131`; top 20ms post high-band
  average `0.00735 -> 0.00377`.
- faster-whisper large-v3-turbo CUDA ASR preserved the transcript:
  `这是普通话测试` / `声音清楚自然`.
