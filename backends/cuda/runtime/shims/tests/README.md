# CUDA AOTI Shim Tests

Unit tests for the CUDA AOTI (Ahead-Of-Time Inductor) shim functions used by the ExecuTorch CUDA backend.

## Prerequisites

1. **CUDA Toolkit**: Ensure CUDA is installed and available
2. **ExecuTorch with CUDA**: Build and install ExecuTorch with CUDA support first

## Building ExecuTorch with CUDA

From the ExecuTorch root directory:

```bash
# Release build
cmake --workflow --preset llm-release-cuda

# Or debug build (recommended for debugging test failures)
cmake --workflow --preset llm-debug-cuda
```

## Building and Run the Tests

### Option 1: Using CMake Presets (Recommended)

From this directory (`backends/cuda/runtime/shims/tests/`):

```bash
# Release build
cmake --workflow --preset default

# Debug build
cmake --workflow --preset debug
```

### Option 2: Manual CMake Commands

From the ExecuTorch root directory:

```bash
# Configure
cmake -B cmake-out/backends/cuda/runtime/shims/tests \
  -S backends/cuda/runtime/shims/tests \
  -DCMAKE_PREFIX_PATH=$(pwd)/cmake-out \
  -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build cmake-out/backends/cuda/runtime/shims/tests -j$(nproc)
```

### Run Specific Test Cases

Use Google Test filters to run specific test cases:

```bash
# From the build directory
cd cmake-out/backends/cuda/runtime/shims/tests
# Run only device mismatch tests
./test_aoti_torch_create_tensor_from_blob_v2 --gtest_filter="*DeviceMismatch*"

# Run a single test
./test_aoti_torch_create_tensor_from_blob_v2 --gtest_filter="AOTITorchCreateTensorFromBlobV2Test.BasicFunctionalityCUDA"

# List all available tests
./test_aoti_torch_create_tensor_from_blob_v2 --gtest_list_tests
```

## Troubleshooting

### CUDA Not Available

If tests are skipped with "CUDA not available", ensure:
- CUDA drivers are installed
- A CUDA-capable GPU is present
- `nvidia-smi` shows the GPU

### Link Errors

If you get link errors, ensure ExecuTorch was built with CUDA support:
```bash
cmake --workflow --preset llm-release-cuda
```

### Test Failures

For debugging test failures, build with debug mode:
```bash
cmake --workflow --preset debug
```

Then run with verbose output:
```bash
./test_aoti_torch_create_tensor_from_blob_v2 --gtest_break_on_failure
```
