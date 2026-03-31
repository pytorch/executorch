# XNNPACK Confidence Status

This note records the measurement fixes we made before starting MLX work and the
best warmed XNNPACK path we can currently defend.

## Measurement fixes completed

- `codegen_ms` now excludes in-loop streaming decode checkpoints. Previously the
  metric double-counted chunk decode time and overstated the hot loop cost.
- Non-streaming `first_audio_ms` is now anchored to request start instead of the
  start of the final decode phase, so it is comparable to streaming runs.
- The CLI now reports `audio` and `rtf` from the raw waveform before silence
  trimming. `trimmed_audio` and `rtf_trimmed` are logged separately so
  post-processing no longer inflates the main throughput metric.

## Best verified warmed XNNPACK path

Use the bounded overlap-window decoder, not the fixed-shape
`decode_audio_stream` surface:

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path <model.pte> \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --prompts_path examples/models/qwen3-tts/benchmark_prompts.txt \
  --repeat 1 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_k 50 \
  --disable_streaming_decoder_surface
```

Warmed prompt-set results after the accounting fixes:

| Export | Max seq len | Avg raw RTF | Avg first audio | Avg codegen | Avg decode |
|--------|-------------|-------------|-----------------|-------------|------------|
| Checked-in unified export | 256 | `0.51x` | `3.57s` | `9.16s` | `1.57s` |
| Tuned experimental export | 160 | `0.52x` | `3.43s` | `8.74s` | `1.61s` |

Notes:

- The `max_seq_len=160` export remains the best measured XNNPACK artifact so far,
  but only by a small margin after the metric fix.
- The checked-in `decode_audio_stream` surface is still slower than the dynamic
  overlap-window fallback on this backend.
- The hot loop is still dominated by talker/code-predictor generation, not audio
  decode.

## Additional non-streaming sanity check

Spot check command:

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path examples/models/qwen3-tts/qwen3_tts_exports_unified/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --text "Hello from the non streaming timing check." \
  --max_new_tokens 64 \
  --temperature 1.0 \
  --top_k 50 \
  --non_streaming_mode \
  --disable_streaming_decoder_surface
```

Observed result:

- `first_audio_ms = 7685.4`
- `generation_ms = 7685.4`
- `final_decode_ms = 902.5`

That matches the expected non-streaming behavior: first audio is only available
after the full generation + final decode path completes.

## What we can now say with confidence

- The metric layer is no longer hiding chunk decode time inside `codegen_ms`.
- The raw XNNPACK throughput number is about `0.51x` to `0.52x` realtime on the
  current warmed short-prompt benchmark.
- The best current XNNPACK path is `--disable_streaming_decoder_surface`.
- XNNPACK streaming is still below realtime after load.

## What still blocks a "faster than mlx-audio" claim

- We still do bounded window re-decode on the decoder side; we do not yet have a
  true stateful incremental vocoder path like the MLX reference.
- The tuned `max_seq_len=160` export is reproducible, but it is not yet the
  default checked-in artifact or a documented export preset.
- We do not yet have an apples-to-apples benchmark harness that runs our future
  MLX path and the upstream `mlx-audio` path on the exact same prompt set.
- `decode_audio_stream` remains a regression on current XNNPACK, so the export
  surface intended for streaming still needs backend-specific tuning.
