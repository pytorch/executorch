# CPU torchvision exports

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
