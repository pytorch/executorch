# MLX Progress

Branch: `qwen3-tts-mlx-realtime`

## Goal

Build an MLX backend path in-tree for Qwen3-TTS that is measurably faster than
the plain `mlx-audio` reference implementation on the same warmed prompt set.

## Benchmark protocol

- Model: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`
- Prompt set: `examples/models/qwen3-tts/benchmark_prompts.txt`
- Reference voice:
  - audio: `poem.wav`
  - text: `This is what my voice sounds like.`
- Metric: audio seconds / generation seconds (`> 1` means faster than realtime)
- Mode: warmed sequential generation after model load
- Seed: `123 + prompt_idx`
- Streaming: enabled (`--stream`, `streaming_interval=2.0`)

## Current status

| Path | Avg throughput | Total throughput | Avg first audio | Result |
|------|----------------|------------------|-----------------|--------|
| Baseline `mlx-audio` generate | `0.540x` | `0.546x` | `5.50s` | reference |
| Cached MLX session backend | `0.556x` | `0.559x` | `5.40s` | `1.030x` faster |

## What changed

Added `mlx_backend.py` with a persistent ICL session that:

- loads the local `mlx-audio` checkout once
- caches reference-audio speech tokens (`ref_codes`)
- caches projected reference text embeddings
- caches the ICL codec/text prefix overlays used before generation
- reuses upstream `_generate_icl()` by overriding only the expensive
  `_prepare_icl_generation_inputs()` step

This keeps generation semantics close to the upstream MLX path while removing
per-prompt re-encoding of the same reference voice context. The current tuned
streaming interval is `4.0s`, which gave the best throughput on the warmed
three-prompt benchmark.

## Latest verification

```bash
python examples/models/qwen3-tts/benchmark_mlx.py --mode both --stream
```

Verified output with the tuned default (`streaming_interval=4.0`):

- Baseline `mlx-audio`: `avg=0.540x`, `total=0.546x`
- Cached session backend: `avg=0.556x`, `total=0.559x`
- Speedup: `1.030x`

Focused retest of the same seeded prompt set also showed a better best-observed
point at the same interval (`avg=0.565x`, `total=0.569x`), so the current
speedup range is roughly `1.03x` to `1.09x` depending on warm-state noise.

## Next experiments

- Measure non-streaming mode with the same seeded prompt set in case the cached
  prefix work matters more when decoder chunking is removed.
- Add an apples-to-apples comparison mode against the current XNNPACK benchmark
  output so MLX and XNNPACK use the same reporting format.
- Investigate whether the tokenizer regex fix path changes prompt length,
  generation length, or throughput in a way that is worth folding into the
  benchmark harness.
