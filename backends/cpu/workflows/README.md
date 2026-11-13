# Torchvision CPU workflows

List classifiers or export a model with the registered CPU or XNNPACK FP32 recipe:

```sh
python -m executorch.backends.cpu.workflows.export --list-models
python -m executorch.backends.cpu.workflows.export \
  --model convnext_tiny --backend cpu --output /tmp/convnext-cpu \
  --source-revision REV
```

Inside fbsource, run `buck2 run fbcode//executorch/backends/cpu:export_torchvision --`
with the same arguments.

The default model is `convnext_tiny` with pretrained weights. Use
`--random-weights` for deterministic random weights, or `--pt2` to reuse a capture.
Input size defaults to the pretrained weights' crop size, or the model default
for random weights; `--image-size` overrides it. For random-weight ViTs, this
also configures the model's input size. Pretrained ViTs reject sizes that do not
match their weights. Reused captures keep their recorded input shape, and the
model name defaults to the capture's file stem.

Exports default to `--runtime et` (`model.pte`), using the registered FP32
`--backend cpu|xnnpack` recipe. `--runtime native --backend cpu` uses `to_native()`
with standard Native preprocessing to produce `model.ptn`. Reuse `--pt2` to compare
the same captured model through the two pipelines.

Both formats include the captured `.pt2`, `input.bin`, complete eager logits in
`eager.bin`, `golden.bin`, and an `export.json` manifest with artifact hashes,
preprocessing provenance, and runtime/artifact identity. `golden.bin` holds the
edge-program logits for ET exports and matches `eager.bin` for Native exports. A
runner can feed `input.bin` to either artifact and compare its output against
`eager.bin` to validate it before measuring latency.

The local CPU runners support repeatable warm-latency checks through
`ET_CPU_WARMUP_RUNS` and `ET_CPU_TIMED_RUNS`. Golden checks use `ET_CPU_ATOL` and
`ET_CPU_RTOL`; both default to `1e-4`.

## ETDump

Every CPU plan step records a delegate event named
`provider/implementation:first_node[+N]`, so all providers are traced without
provider changes. The XNNPACK provider also enables XNNPACK runtime profiling and
records `<operator> #<n>` events, such as `Transpose (ND, X32) #3`, inside its
step, with the operator's input and output shapes as JSON delegate metadata
(`shapes` in `profile.json`).
Tracing compiles out unless the runner is built with the event tracer enabled,
and the shapes, which read XNNPACK internals, also need
`executorch.cpu_xnnpack_operator_shapes`. Inside fbsource, build the traced
runner and the workflow, then pass the two output paths that Buck prints:

```bash
buck build -j 2 @fbcode//mode/opt -c executorch.event_tracer_enabled=true -c executorch.cpu_xnnpack_operator_shapes=true fbcode//executorch/backends/cpu:runner fbcode//executorch/backends/cpu:etdump_cpu --show-full-json-output
CPU_TRACED_RUNNER=<path of :runner>
CPU_ETDUMP=<path of :etdump_cpu>
"$CPU_ETDUMP" --runner "$CPU_TRACED_RUNNER" --model artifacts/cpu/model.pte --golden artifacts/cpu/golden.bin --output results/etdump
```

`:runner` links only XNNPACK and ET, so XNNPACK takes the whole ConvNeXt graph.
It has no CMake target.

The runner writes `model.etdump` when `ET_CPU_ETDUMP` is set, and logs each CPU
delegate's plan, including its step event names, at `Info`, so it must be built
with logging enabled. The workflow drops warmup samples and writes
`profile.json` with median/min/max per event. Events from method loading are
reported as `init`. Events inside a CPU delegate call carry the index of that
delegate's plan in `plan`, so operators with the same name from different
delegates stay distinct. Provider events, which nest inside their step, include
XNNPACK operators and Reference's `native_call_<op>` kernel events, so step and
provider totals are reported separately. The workflow fails if any plan step,
or any XNNPACK delegate's operator events, are missing, or if an executed event
lacks a sample for some run. Trailing positional arguments are passed to the
runner as provider preferences.
