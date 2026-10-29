# CPU torchvision exports

List classifiers or export a PTE with the registered CPU or XNNPACK FP32 recipe:

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
Exports include `model.pte`, the captured `.pt2`, `input.bin`, `eager.bin`,
`golden.bin`, and an `export.json` manifest with artifact hashes and preprocessing
provenance. `input.bin` is the raw float32 input; `eager.bin` and `golden.bin`
hold the eager and edge-program logits. A runner can feed `input.bin` to
`model.pte` and compare its output against `eager.bin` to validate the PTE before
measuring latency.
