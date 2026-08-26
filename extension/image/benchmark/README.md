# ImageProcessor benchmark

A microbenchmark for the `ImageProcessor` reuse APIs (`process_into` and
`process_yuv_into`) plus a companion script to compare two runs.

## What it measures

`image_processor_benchmark` sweeps common input sizes × target sizes and, per
cell, times a set of variants:

- **API**: `process_into` (BGRA/RGBA) and `process_yuv_into` (NV12/NV21)
- **execution path**: CPU, GPU, and the size-threshold default
- **resize mode**: stretch, letterbox
- **orientation**: upright and 90° rotate
- **other**: cropped ROI, and the allocating `process()` vs `process_into()`

Each row reports mean / median / p95 / stddev over 100 measured iterations
(10 warmup).

## Build mode matters

Always benchmark an **optimized** build. The default `buck2 run` compiles at
`-O0`, where the hand-written NEON kernels are unrepresentative. Pass `-c cxx.extra_cxxflags=-Os` to match
how ExecuTorch ships:

```bash
buck2 run -c cxx.extra_cxxflags=-Os \
  fbsource//xplat/executorch/extension/image/benchmark:image_processor_benchmark
```

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--format=bgra\|rgba\|nv12\|nv21` | all | restrict to one color / YUV format |
| `--unit=cpu\|gpu\|default` | all | restrict to one execution path |
| `--out=PATH` | stdout | write the results table to PATH |

The input-size sweep and the rotation variant always run. Writing with `--out`
keeps the file free of buck build-log lines (which go to stderr).

## Comparing two runs

Capture a baseline and a candidate, then diff them:

```bash
TARGET=fbsource//xplat/executorch/extension/image/benchmark:image_processor_benchmark
buck2 run -c cxx.extra_cxxflags=-Os $TARGET -- --out=/tmp/base.txt
# ... make your change ...
buck2 run -c cxx.extra_cxxflags=-Os $TARGET -- --out=/tmp/new.txt

python3 xplat/executorch/extension/image/benchmark/compare_benchmarks.py \
  /tmp/base.txt /tmp/new.txt
# or via buck:
buck2 run fbsource//xplat/executorch/extension/image/benchmark:compare_benchmarks \
  -- /tmp/base.txt /tmp/new.txt
```

`compare_benchmarks.py` matches rows by (API section, input→target cell, variant)
and prints the per-row `base / new` speedup plus a summary bucketed by execution
path (CPU / GPU / default). Cross-run and thermal drift shift all rows together,
so compare the buckets against each other rather than reading any single ratio
absolutely.

For a clean A/B, capture both files back-to-back on an otherwise idle machine.

## Files

- `image_processor_benchmark.cpp` — the benchmark binary; buck target
  `:image_processor_benchmark` (run with `buck2 run`)
- `compare_benchmarks.py` — compares two result files (stdlib only); buck target
  `:compare_benchmarks` (run with `buck2 run …:compare_benchmarks -- BASE NEW`)
- `BUCK` / `TARGETS` / `targets.bzl` — build definitions
