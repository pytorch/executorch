# AMD ROCm support (experimental)

The AOTI CUDA backend can be built against AMD ROCm/HIP. PyTorch exposes AMD
GPUs as device type `cuda`, so names such as `CudaBackend` and
`aoti_cuda_blob.ptd` remain unchanged.

`EXECUTORCH_BUILD_ROCM` is off by default, is never auto-enabled, and is
mutually exclusive with `EXECUTORCH_BUILD_CUDA`. It requires
`EXECUTORCH_BUILD_EXTENSION_TENSOR`. Execution has been validated only on
MI300X (`gfx942`).

## Requirements and limitations

- Supply a ROCm PyTorch build and matching Triton AMD backend. To avoid replacing
  ROCm PyTorch with a CPU wheel, install ExecuTorch with
  `pip install -e . --no-build-isolation`.
- CUDA graphs and CUDA-only fallback shims are unavailable.
- Python pybindings cannot allocate ROCm device memory; use a native runner such
  as `executor_runner`.
- Installed CMake consumers must be able to find the HIP package.
- Model runners do not yet link the ROCm backend, and there is no ROCm CI.

## Build

```bash
cmake -S . -B cmake-out-rocm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');/opt/rocm" \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_ROCM=ON
cmake --build cmake-out-rocm --target aoti_cuda_backend aoti_cuda_shims -j
```

Verify that the build uses HIP and not the NVIDIA runtime:

```bash
ldd cmake-out-rocm/backends/cuda/libaoti_cuda_shims.so | grep -E 'amdhip64|cudart'
```

A `libcudart` dependency is a configuration error.
