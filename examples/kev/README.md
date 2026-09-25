# Native Kev decision inference

Embed [Kev](https://huggingface.co/jaredpalmer/kev-0.8b) in a C++ application
with ExecuTorch's `Module` API, using XNNPACK on CPU or MLX on Apple GPUs.
Prefill a shared text once, then ask batches of questions from that snapshot.
Inference uses the native binary, model, and tokenizer; Python is needed for
export.

[main.cpp](main.cpp) is the complete application. It loads the model and
tokenizer and asks three questions about a customer message in one
`system_one` call. The output contains a Choice answer with its distribution
and two Noul probabilities. [benchmark.cpp](benchmark.cpp) measures explicit
prefix reuse across two evaluation calls.

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

The default limits are 384 prefix tokens and 1,024 tokens for the prefix plus
one question. Set larger export bounds for longer documents:

```bash
PYTHONPATH="$HOME/kev:$PYTHONPATH" python examples/kev/export.py \
  --checkpoint kev-checkpoint --backend mlx --dtype bf16 --output kev-mlx-long \
  --max-prefix 2048 --max-context 4096
```

Token counts include delimiters. Larger bounds increase planned memory;
exceeding a bound at inference returns an error without truncating the input.
Longer contexts were not part of the checkpoint's training configuration.

Constant methods record these bounds and the tokenizer IDs. `get_temperature`
returns the fitted temperature as a float; `get_checkpoint_id` returns
`sha256:<digest>` of the exported checkpoint's `head.pt` bytes. Call them through
`Module::execute` to check the program against the checkpoint used by the app.

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
  -DEXECUTORCH_BUILD_MLX=ON
cmake --build cmake-out/kev-mlx --target kev_runner --parallel 8

cmake-out/kev-mlx/kev_runner kev-mlx/model.pte kev-mlx/tokenizer.json
```

Fresh MLX builds default to a 26.2 deployment target on macOS 26.2+ and 14.0
on older systems. With macOS SDK 26.2+ and Metal 4, the 26.2 target includes
MLX's NAX matrix kernels for M5; the same binary uses standard kernels on M1–M4.
To ship a build from a newer Mac to macOS 14+, set
`-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0`; this excludes NAX kernels.
An explicit target, including `MACOSX_DEPLOYMENT_TARGET` in the environment,
takes precedence. Existing CMake build directories retain their cached target;
pass `-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2` to update a previous 14.0 build.

CMake copies `mlx.metallib` beside each MLX executable; ship it with the binary.
The optional `TEXT` argument replaces the bundled customer message.

The current tokenizer revision can drop accents during NFC normalization.
The bundled ASCII request is unaffected; the NFC correction is being handled
separately in the tokenizer library.

## Benchmark

Build and run `kev_benchmark` to measure the same three questions split across
two evaluation calls sharing one prefix:

```bash
cmake --build cmake-out/kev-cpu --target kev_benchmark --parallel 8
cmake-out/kev-cpu/kev_benchmark kev-cpu/model.pte kev-cpu/tokenizer.json

cmake --build cmake-out/kev-mlx --target kev_benchmark --parallel 8
cmake-out/kev-mlx/kev_benchmark kev-mlx/model.pte kev-mlx/tokenizer.json
```

The benchmark loads the model once, performs two warmups, then reports median
milliseconds over five runs for prefill, cached evaluation, and prefill plus
evaluation. Each run creates a prefix and reuses it for both evaluation calls.
Timing includes tokenization and prefix snapshot creation, excludes loading and
printing, and waits for GPU results. An optional third argument replaces the
bundled text.

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
To reproduce the XNNPACK BF16 column with the CPU benchmark:

```bash
PYTHONPATH="$HOME/kev:$PYTHONPATH" python examples/kev/export.py \
  --checkpoint kev-checkpoint --backend xnnpack --dtype bf16 --output kev-cpu-bf16

cmake-out/kev-cpu/kev_benchmark kev-cpu-bf16/model.pte kev-cpu-bf16/tokenizer.json
```

## C++ API

[api.h](api.h) defines the question and answer types and the virtual
`SystemOne::system_one(state, questions)` interface, following
[TypeSafe's SDK operation](https://docs.typesafe.ai/sdk/python/api/clients/sync).
It returns ExecuTorch's `Result<Answers>`. [kev.h](kev.h) provides `Kev`,
which implements this interface and borrows a `Module` and tokenizer.

For explicit prefix reuse, `Kev` also provides `prefill(state)` and
`evaluate(prefix, questions)`. `Prefix` owns its snapshot and must be used with
the `Kev` instance that created it. The instance must outlive its prefixes, and
the `Module` and tokenizer must outlive the instance. Multiple prefixes may
coexist; all calls must be serialized per `Module`.

Each question starts from the same snapshot. Evaluation leaves it unchanged,
so later calls can ask different questions. Answers own their labels and scores.

The question and answer types follow [TypeSafe's typed SDK](https://docs.typesafe.ai/sdk/python/api/types/questions).
`kev.cpp` handles Kev's input encoding and answer mapping. `Question` is a
`std::variant<Choice, Noul, Score>`; `Answer` holds the matching answer type.
`Questions` and `Answers` are ordered `(id, value)` pairs. IDs identify answers
and are not sent to the model.

```cpp
kev::Questions questions{
    {"department", kev::Choice{"Which team should handle this?",
        {{"billing", "Invoices and refunds"}, {"technical", "Bugs and outages"}}}},
    {"refund", kev::Noul{"Is a refund requested?",
        {{kev::NoulOutcome::False, "No refund requested"},
         {kev::NoulOutcome::True, "Explicitly asks for a refund"}}}},
    {"urgency", kev::Score{"How urgent is this?",
        {"Can wait", "This week", "Today"}}}};
```

With a loaded module and tokenizer, call through the interface:

```cpp
kev::Kev model(module, tokenizer);
kev::SystemOne& api = model;
auto answers = api.system_one(state, questions);
```

Each `system_one` call prefills the supplied state and evaluates its questions.
For multiple requests about the same state, reuse a prefix:

```cpp
auto prefix = model.prefill(state);
if (!prefix.ok()) {
  return 1;
}
auto answers = model.evaluate(*prefix, questions);
if (!answers.ok()) {
  return 1;
}
auto followup = model.evaluate(*prefix, {
    {"duplicate", kev::Noul{"Was the customer charged more than once?", {}}}});
if (!followup.ok()) {
  return 1;
}
```

Both calls use the same unchanged snapshot. See [benchmark.cpp](benchmark.cpp)
for a complete example with timing.

Text fields accept UTF-8 strings; structured content can be rendered to text
before calling this native interface. Instructions may be empty. Choice
criteria preserve caller order and have unique names with optional descriptions
(`std::nullopt` for none). Noul criteria are optional descriptions keyed by
`NoulOutcome::False` and `NoulOutcome::True`, corresponding to no and yes.
Score criteria are ordered descriptions: repeated and empty levels are allowed
and retain their separate indices.

`ChoiceAnswer` contains `choice`, `probabilities` by name, and `confidence`.
`NoulAnswer::noul` is the probability of yes. `ScoreAnswer` contains the expected
zero-based `score`, ordered `legend` and `probabilities` vectors, and
`confidence`. Each vector index identifies the corresponding level in
`Score::criteria`; for example, `answer.probabilities.at(0)` gives the first
level's probability. Values retain full precision. Confidence uses Kev's
formulas; exact equivalence to TypeSafe is not established.

`evaluate` accepts any nonempty list of questions with unique IDs and splits it
into batches of up to eight, reusing the same prefix. Answers retain request
order. Each batch is padded to its own longest question and criterion count.
Choice and Score accept 1–255 criteria, following upstream Kev; TypeSafe's
hosted API documents 2–10 Score levels.
