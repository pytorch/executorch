# Native Kev decision inference

Embed [Kev](https://huggingface.co/jaredpalmer/kev-0.8b) in a C++ application
with ExecuTorch's `Module` API, using XNNPACK on CPU or MLX on Apple GPUs.
Prefill a shared text once, then ask batches of questions from that snapshot.
Inference uses the native binary, model, and tokenizer; Python is needed for
export.

[main.cpp](main.cpp) is the complete application. It loads the model and
tokenizer, prefills a customer message, and asks three questions in two calls.
The output contains selected labels and probabilities for each option.

## Export

Use an ExecuTorch development environment and a local `~/kev` checkout at
[d3d2f73](https://github.com/jaredpalmer/kev/tree/d3d2f73cc5f7828ae9c114c3d221ef454801b903).
This example uses Kev's loader to merge LoRA in FP32, then casts the backbone to
the requested precision. The pointer head, temperature scaling, and DeltaNet
recurrence stay in FP32. No weight quantization is applied.

```bash
python -m pip install transformers==5.17.0 peft==0.21.0

hf download jaredpalmer/kev-0.8b \
  --revision 54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8 \
  --local-dir kev-checkpoint

PYTHONPATH="$HOME/kev:$PYTHONPATH" python examples/kev/export.py \
  --checkpoint kev-checkpoint --backend xnnpack --dtype fp32 --output kev-cpu

PYTHONPATH="$HOME/kev:$PYTHONPATH" python examples/kev/export.py \
  --checkpoint kev-checkpoint --backend mlx --dtype bf16 --output kev-mlx
```

Both backends support `fp32` and `bf16`. Each export contains `model.pte` and its
tokenizer files. The program has two methods: `prefill` returns convolution
history, DeltaNet state, and attention KV; `score` fans that prefix out across
question rows and applies the pointer head to option-boundary hidden states.
It preserves the checkpoint's fitted temperature and has no generation loop.

## Build and run

From the ExecuTorch root, with submodules initialized:

```bash
cmake -S examples/kev -B cmake-out/kev-cpu \
  -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build cmake-out/kev-cpu --target kev_runner --parallel 8

cmake-out/kev-cpu/kev_runner kev-cpu/model.pte kev-cpu/tokenizer.json
```

For MLX, use macOS 14+ with Xcode's Metal compiler installed:

```bash
cmake -S examples/kev -B cmake-out/kev-mlx \
  -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DEXECUTORCH_BUILD_MLX=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build cmake-out/kev-mlx --target kev_runner --parallel 8

cmake-out/kev-mlx/kev_runner kev-mlx/model.pte kev-mlx/tokenizer.json
```

CMake copies `mlx.metallib` beside the MLX runner; ship it with the binary. The
optional `TEXT` argument replaces the bundled customer message.

The current tokenizer revision can drop accents during NFC normalization.
The bundled ASCII request is unaffected; the NFC correction is being handled
separately in the tokenizer library.

## Benchmark

Append `--benchmark` to measure the same three questions in two evaluation
calls:

```bash
cmake-out/kev-cpu/kev_runner kev-cpu/model.pte kev-cpu/tokenizer.json --benchmark
cmake-out/kev-mlx/kev_runner kev-mlx/model.pte kev-mlx/tokenizer.json --benchmark
```

The runner loads the model once, performs two warmups, then reports median
milliseconds over five runs for prefill, cached evaluation, and prefill plus
evaluation. Each run creates a prefix and reuses it for both evaluation calls.
Timing includes tokenization and prefix snapshot creation, excludes loading and
printing, and waits for GPU results. To use different text, put it before
`--benchmark`.

Measurements on an Apple M1 Pro (8 performance cores, 2 efficiency cores,
32 GiB RAM), macOS 26.6.2, using a Release build with Apple Clang 17 and the
default thread pool. Measured on September 22, 2026, with the pinned checkpoint
above and the bundled customer message (19 prefix tokens including the state
delimiter). Cached evaluation covers all three questions across both calls.

| Median latency (ms) | XNNPACK FP32 | XNNPACK BF16 | MLX BF16 |
|---|---:|---:|---:|
| Prefill | 100.28 | 176.45 | 31.55 |
| Cached evaluation | 457.59 | 804.16 | 86.55 |
| Prefill + evaluation | 562.93 | 980.17 | 118.56 |

The combined value is the median of complete runs, so it need not equal the sum
of the other medians. FP32 is faster on this CPU; all exports are unquantized.
To reproduce the XNNPACK BF16 column with the CPU runner:

```bash
PYTHONPATH="$HOME/kev:$PYTHONPATH" python examples/kev/export.py \
  --checkpoint kev-checkpoint --backend xnnpack --dtype bf16 --output kev-cpu-bf16

cmake-out/kev-cpu/kev_runner kev-cpu-bf16/model.pte kev-cpu-bf16/tokenizer.json --benchmark
```

## C++ API

[kev.h](kev.h) exposes `prefill(module, tokenizer, state)` and
`evaluate(prefix, questions)`, both returning `Result`. `Prefix` owns its
snapshot and borrows the `Module` and tokenizer, which must outlive it. Multiple
prefixes may coexist; calls must be serialized per `Module`.

Each question starts from the same snapshot. Evaluation leaves it unchanged,
so later calls can ask different questions. Answers own their labels and scores.

The export supports 1–8 questions per call, 1–255 options per question, 384 prefix
tokens, and 1,024 tokens for prefix plus each branch. Overlong requests fail
without truncation. Question IDs and option labels must be unique; instructions
and labels must be nonempty.
