# Qwen3-TTS No-Compromise Backlog

This file tracks work we are explicitly deferring while the current milestone
focuses on text-only end-to-end C++ synthesis through
`qwen3_tts_unified_runner`.

## Must-Fix Semantic Gaps

- [ ] Fix the talker/codebook token-space mismatch so `codec_eos_id` is actually reachable from the sampled talker vocabulary instead of relying on `max_new_tokens`.
- [ ] Remove the decoder clamp fallback that silently maps out-of-range codec ids to `0`.
- [ ] Close the remaining text-generation parity gap with `generate_codes.py` for `language`, `top_p`, `repetition_penalty`, and `non_streaming_mode`.
- [ ] Verify that the assistant-wrapped prompt string used by the C++ runner matches the upstream `qwen_tts` helper exactly, not just the `mlx-audio` approximation.
- [ ] Add a deterministic parity harness that compares text-only codec traces between the Python helper and the unified C++ path.

## Deferred Feature Parity

- [ ] Internalize `ref_audio` + `ref_text` voice-clone prompting into the unified C++ runner.
- [ ] Support x-vector-only speaker prompting in the unified C++ path.
- [ ] Support full ICL prompting with reference speech-token context instead of text-only prompting.
- [ ] Decide whether to export extra primitives for speaker-conditioning flows or keep them in host-side orchestration.
- [ ] Add explicit support for non-English language conditioning instead of logging and falling back to the text-only default.

## Performance And Realtime Work

- [ ] Measure the new text-only unified C++ path end to end and compare it against the two-stage `generate_codes.py -> qwen3_tts_unified_runner` baseline.
- [ ] Build a proper realtime scorecard: cold start, warm start, first-audio latency, full decode latency, and realtime factor.
- [ ] Optimize the code-predictor path, which still dominates projected CPU latency.
- [ ] Explore whether the `cp_generate` fused path can reduce per-step latency further without changing semantics.
- [ ] Revisit streaming decode with MLX-style persistent conv-buffer reuse instead of chunked re-decode.
- [ ] Revisit Metal/GPU acceleration once the text-only C++ semantics are stable.

## Packaging And Reproducibility

- [ ] Re-export the unified `.pte` artifacts after the prompt-contract metadata changes and verify they load correctly.
- [ ] Package `tokenizer.json` alongside unified export artifacts or define a stable artifact-discovery contract for text mode.
- [ ] Audit all checked-in unified export manifests and generated artifacts for drift against current source.
- [ ] Decide whether checked-in `.pte` artifacts should remain in-tree or move to reproducible export scripts plus manifests only.
- [ ] Capture the exact export command and expected artifact set for the unified text-only path.

## Validation And Regression Coverage

- [ ] Add a regression test that fails if a checked-in manifest drops `cp_generate` again.
- [ ] Add a stronger runner-contract test that validates text-mode CLI behavior against the built binary, not just source text.
- [ ] Add export-metadata tests for the new prompt-budget constant methods.
- [ ] Add end-to-end smoke coverage for at least one short prompt and one longer prompt through the unified C++ path.
- [ ] Add explicit coverage for prompt-budget failure cases: prompt too short, `max_new_tokens` too small, and `max_seq_len` overflow.
- [ ] Add a reproducible fixture corpus for text-only, multilingual, x-vector-only, and full clone cases.

## MLX Reference Follow-Ups

- [ ] Decide which `mlx-audio` prompt-preparation pieces should be mirrored directly and which should remain reference-only.
- [ ] Investigate whether MLX's “first text token folded into prefill” path remains correct under the exact upstream `qwen_tts` tokenizer output.
- [ ] Investigate whether MLX's non-streaming ICL overlay should become the long-term reference for ExecuTorch clone mode.
- [ ] Avoid porting MLX runtime-only details such as cache/eval mechanics and heuristic streaming constants without an ExecuTorch-specific rationale.
