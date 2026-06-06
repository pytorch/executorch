# VGF Host Runtime (executor_runner)

This flow runs the VGF-exported `.pte` on host using the portable
`executor_runner` binary built at the repo root. The runtime helper script
serializes the input image, invokes `executor_runner`, then reconstructs the
output image from the generated tensor bytes.

For the smallest reproducible demo in this repo, first create the fixed
text-heavy input and the small LR/HR export set:

```bash
python examples/arm/super_resolution_example_vgf/model_export/prepare_demo_assets.py \
  --output-dir ./demo_assets
```

1. Install ML SDK dependencies and set up the environment:

```bash
examples/arm/setup.sh --disable-ethos-u-deps --enable-mlsdk-deps
source examples/arm/arm-scratch/setup_path.sh
```

2. Build the runner with VGF enabled:

```bash
cmake -B cmake-out \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  .

cmake --build cmake-out --target executor_runner
```

3. Run the exported model on an image that matches the static export size:

```bash
python examples/arm/super_resolution_example_vgf/runtime/run_super_resolution.py \
  --model-path ./demo_assets/swin2sr_x2_vgf_fp.pte \
  --input-image ./demo_assets/runtime/demo_lr_64.png \
  --output-image ./demo_assets/runtime/demo_fp_128.png
```

The runtime helper reads the metadata emitted by the exporter to reconstruct the
output tensor and save it as an image. Because the export uses static shapes,
the input image must match the exported low-resolution dimensions exactly.

Use the same runtime input with `./demo_assets/swin2sr_x2_vgf_int8.pte` to
validate the INT8 path once the quantized export has been generated. If the
host VKML emulation layer rejects quantized shaders, rerun that runtime step on
Linux or on the target Vulkan driver stack.
