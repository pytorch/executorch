# AMD ROCm support (experimental)

The AOTI CUDA backend can be built against AMD ROCm/HIP. PyTorch exposes AMD
GPUs as device type `cuda`, so names such as `CudaBackend` and
`aoti_cuda_blob.ptd` remain unchanged.

`EXECUTORCH_BUILD_ROCM` is off by default, is never auto-enabled, and is
mutually exclusive with `EXECUTORCH_BUILD_CUDA`. It requires
`EXECUTORCH_BUILD_EXTENSION_TENSOR`. Execution has been validated only on
MI300X (`gfx942`); CI is configured to exercise `gfx950`.

## Requirements and limitations

- Supply a ROCm PyTorch build and matching Triton AMD backend. To avoid replacing
  ROCm PyTorch with a CPU wheel, install ExecuTorch with
  `pip install -e . --no-build-isolation`.
- CUDA graphs and CUDA-only fallback shims are unavailable.
- Python pybindings cannot allocate ROCm device memory; use a native runner such
  as `executor_runner`.
- Installed CMake consumers must be able to find the HIP package.
- Voxtral Realtime has an explicit ROCm workflow. Other model runners do not
  claim ROCm support. The gfx950 CI job covers the backend build, native
  runtime, AOTI export and execution, and Triton W4 tests.

## Build

```bash
cmake --workflow --preset llm-release-rocm
```

Verify that the build uses HIP and not the NVIDIA runtime:

```bash
ldd cmake-out-rocm-llm/lib/libaoti_cuda_shims.so | grep -E 'amdhip64|cudart'
```

A `libcudart` dependency is a configuration error.

For an end-to-end export that proves Inductor emitted a Triton kernel, see
[../../examples/cuda/README.md](../../examples/cuda/README.md).

Voxtral Realtime additionally covers BF16 and packed W4/BF16 execution. See
[../../examples/models/voxtral_realtime/README.md](../../examples/models/voxtral_realtime/README.md)
and its `run_rocm_e2e.sh` example.
