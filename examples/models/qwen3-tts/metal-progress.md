# Metal Streaming Progress

Branch: `qwen3-tts-metal-streaming`

## Goal

Build the best practical Metal-backed Qwen3-TTS streaming path in the C++
runner first, using a hybrid deployment where text generation runs on Metal and
the vocoder remains on XNNPACK until a true Metal decoder path is proven.

## Transferred lessons

### From XNNPACK

- The fixed-shape `decode_audio_stream` export is functionally correct, but its
  performance is much more sensitive to emit interval and warm state than the
  overlap-window `decode_audio` fallback.
- The metric layer now separates `codegen_ms`, `first_audio_ms`, and raw
  realtime factor correctly, so we should reuse the same benchmark discipline
  for Metal.

### From MLX

- Stateful decoder ideas are important long term, but the first shipping win is
  often in orchestration and cached context rather than a wholesale model
  rewrite.
- Streaming policy must be validated on a warmed prompt set instead of inferred
  from isolated decode-only timings.

### From `voxtral_realtime`

- Export metadata should act as a runtime contract.
- Backend choice should be represented explicitly instead of inferred loosely in
  the runner.
- Streaming runners should prefer the backend-specific fast path by default, not
  just whichever method happens to exist in the export.

## Current implementation slice

- Added backend split metadata to unified exports:
  - `generation_backend_code`
  - `decoder_backend_code`
  - `prefer_streaming_decoder_surface`
- For current Metal exports, the intended contract is:
  - generation backend = `metal`
  - decoder backend = `xnnpack`
  - preferred streaming decoder surface = `overlap_window`
- Updated the C++ runner to honor that metadata by default while still exposing
  a force flag for experiments:
  - `--force_streaming_decoder_surface`
- Kept `cp_generate` on XNNPACK for Metal exports after confirming that the
  fused method still needs `topk` and `cumsum` fallback kernels that the
  current AOTI Metal backend does not provide.
- Reused the existing Llama MPS fix for bool causal masks by applying
  `replace_causal_mask()` to the Metal-exported talker and code-predictor
  transformers before export.

## Why this is the right first step

Today Qwen3-TTS is not blocked on "no Metal support at all." It already has a
mixed Metal/XNNPACK export path. The real practical issue is that the runner can
still auto-select the slower streaming decoder surface because it only checks
capability, not backend-aware preference.

Fixing that gives us a better hybrid shipping path immediately and makes the
next benchmark meaningful.

## Verification completed

Focused contract tests:

```bash
conda run -n executorch python -m unittest \
  examples.models.qwen3-tts.tests.test_unified_runner_contract \
  examples.models.qwen3-tts.tests.test_unified_quality_contract \
  examples.models.qwen3-tts.tests.test_unified_metadata
```

Result: `PASS`

Runner rebuild:

```bash
cmake --build cmake-out/examples/models/qwen3-tts --target qwen3_tts_unified_runner
```

Result: `PASS`

## Verification completed on Metal artifact

Metal export with the new metadata:

```bash
conda run -n executorch python examples/models/qwen3-tts/export_unified.py \
  --converted-dir examples/models/qwen3-tts/qwen3_tts_artifacts \
  --talker-dir examples/models/qwen3-tts/qwen3_tts_artifacts/talker_converted \
  --output-dir /tmp/qwen3_tts_exports_metal_streaming_maskfix \
  --backend metal \
  --dtype fp32
```

Result: `PASS`

First export attempt failed before lowering completed because `cp_generate` was
still being partitioned to Metal:

```text
RuntimeError: Method cp_generate missing fallback kernels (2 total):
  - at::_ops::cumsum::call
  - at::_ops::topk::call
```

That failure is now the documented reason the branch keeps `cp_generate` on
XNNPACK for the hybrid Metal path.

First runtime attempt with the saved Metal artifact exposed the next backend
compatibility issue:

```text
Unsupported dtype: 11. Supported dtypes: 0 (uint8), 4 (int64), 6 (float32), 15 (bfloat16)
```

Root-cause investigation points to the bool causal mask buffer inherited from
the reused Llama attention stack. The current branch now mirrors the working
Llama MPS path and rewrites those masks to float additive masks before Metal
export.

## Benchmark caveats discovered

- Process-to-process warm state matters a lot on the hybrid Metal artifact even
  after the runner's in-process warmup. The same overlap-window benchmark
  improved from weighted `RTF=0.0658x` on the first process to `RTF=0.1152x` on
  the next process.
- The fixed-surface path is very sensitive to emit cadence. With the same
  artifact, `--streaming_interval 2.0` dropped the weighted prompt-set RTF to
  `0.0528x`, while `--streaming_interval 4.0` raised it to `0.1010x` on a warm
  process.
- Because of those two effects, a single run is not a trustworthy policy signal
  for the Metal branch. We should compare warmed prompt-set runs with matched
  intervals before changing the export default.

## Benchmark commands

Use the same artifact and the same emit interval for both policies:

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path /tmp/qwen3_tts_exports_metal_streaming_maskfix/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --prompts_path examples/models/qwen3-tts/benchmark_prompts.txt \
  --repeat 1 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_k 50 \
  --streaming_interval 4.0
```

```bash
cmake-out/examples/models/qwen3-tts/qwen3_tts_unified_runner \
  --model_path /tmp/qwen3_tts_exports_metal_streaming_maskfix/model.pte \
  --tokenizer_path examples/models/qwen3-tts/qwen3-tts-12Hz-0.6B-Base/tokenizer.json \
  --prompts_path examples/models/qwen3-tts/benchmark_prompts.txt \
  --repeat 1 \
  --max_new_tokens 128 \
  --temperature 1.0 \
  --top_k 50 \
  --streaming_interval 4.0 \
  --force_streaming_decoder_surface
```

If you want numbers that are comparable to the warm-run table below, do one
throwaway process first and judge the second process, not the very first launch
after export.

## Benchmark results

| Run | Policy | Interval | Weighted RTF | Avg first audio | Notes |
| --- | --- | --- | --- | --- | --- |
| First matched pair | overlap-window | 4.0s | `0.0658x` | `49.09s` | Fresh process after export |
| First matched pair | fixed-surface | 4.0s | `0.0699x` | `45.04s` | Fresh process after export |
| Sensitivity probe | fixed-surface | 2.0s | `0.0528x` | `30.22s` | Earlier unfair comparison; included here only to show interval sensitivity |
| Warm rerun | overlap-window | 4.0s | `0.1152x` | `28.97s` | Best current apples-to-apples result |
| Warm rerun | fixed-surface | 4.0s | `0.1010x` | `30.70s` | Still useful for short utterances, but worse overall |

Warm-run prompt details:

- Overlap-window:
  - prompt 0: `audio=1.68s`, `generation=20.30s`, `rtf=0.08x`
  - prompt 1: `audio=6.16s`, `generation=52.40s`, `rtf=0.12x`
  - prompt 2: `audio=8.08s`, `generation=65.49s`, `rtf=0.12x`
- Fixed-surface:
  - prompt 0: `audio=1.68s`, `generation=18.50s`, `rtf=0.09x`
  - prompt 1: `audio=6.16s`, `generation=58.43s`, `rtf=0.11x`
  - prompt 2: `audio=8.08s`, `generation=80.74s`, `rtf=0.10x`

## Current decision

- Keep `prefer_streaming_decoder_surface = 0` for the current hybrid Metal
  export. On the warmed apples-to-apples `4.0s` benchmark, overlap-window beats
  fixed-surface on weighted prompt-set throughput (`0.1152x` vs `0.1010x`) and
  average first-audio latency (`28.97s` vs `30.70s`).
- Keep `--force_streaming_decoder_surface` as an experiment knob. It can still
  help on very short utterances, and it is the right path to compare when we
  revisit a true Metal decoder surface later.
- Treat `--streaming_interval 4.0` as the current benchmark baseline for this
  branch. `2.0s` is too punitive to the fixed-surface path and obscures the real
  policy decision.
- Any future claims about Metal streaming speed should use a warmed prompt-set
  benchmark and call out whether the result comes from the first or second
  process after export.

## Next decision gate

The runner/export contract is now stable enough to move to second-stage tuning:

- explain the large cross-process warm-state delta on the Metal artifact
- benchmark interval and chunk-size tuning around the overlap-window default
- evaluate whether a deeper hybrid pipeline overlap change is worth the added
  complexity
- defer true Metal vocoder work until the hybrid baseline is clearly established
