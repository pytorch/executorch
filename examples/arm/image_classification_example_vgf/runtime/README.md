# VGF Host Runtime (executor_runner)

This flow runs the VGF-exported `.pte` on host using the portable
`executor_runner` binary built at the repo root.

1. Install MLSDK dependencies and set up the environment:

```bash
examples/arm/setup.sh --disable-ethos-u-deps --enable-mlsdk-deps
source examples/arm/arm-scratch/setup_path.sh
```

2. Build the runner with VGF enabled:

```bash
cmake -B cmake-out \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_VGF=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  .

cmake --build cmake-out --target executor_runner
```

3. Run the exported model:

```bash
./cmake-out/executor_runner -model_path ./deit_quantized_vgf.pte
```
