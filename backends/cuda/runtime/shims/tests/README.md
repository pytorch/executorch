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

## Building the Tests

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

## Running the Tests

### Run All Tests

```bash
# Using ctest (from the build directory)
cd cmake-out/backends/cuda/runtime/shims/tests
ctest --output-on-failure

# Or using the test preset (from this directory)
ctest --preset default
```

### Run a Specific Test

```bash
# From the build directory
./test_aoti_torch_create_tensor_from_blob_v2
./test_aoti_torch_empty_strided
./test_aoti_torch_delete_tensor_object
./test_aoti_torch_copy_
./test_aoti_torch_new_tensor_handle
./test_aoti_torch_item_bool
./test_aoti_torch_assign_tensors_out
```

### Run Specific Test Cases

Use Google Test filters to run specific test cases:

```bash
# Run only device mismatch tests
./test_aoti_torch_create_tensor_from_blob_v2 --gtest_filter="*DeviceMismatch*"

# Run a single test
./test_aoti_torch_create_tensor_from_blob_v2 --gtest_filter="AOTITorchCreateTensorFromBlobV2Test.BasicFunctionalityCUDA"

# List all available tests
./test_aoti_torch_create_tensor_from_blob_v2 --gtest_list_tests
```

## Test Descriptions

| Test File | Description |
|-----------|-------------|
| `test_aoti_torch_create_tensor_from_blob_v2` | Tests tensor creation from existing memory blobs, including device type validation |
| `test_aoti_torch_empty_strided` | Tests creation of uninitialized tensors with specified strides |
| `test_aoti_torch_delete_tensor_object` | Tests proper tensor deletion and memory management |
| `test_aoti_torch__reinterpret_tensor` | Tests tensor view reinterpretation with different shapes/strides |
| `test_aoti_torch_copy_` | Tests tensor copy operations between CPU and CUDA |
| `test_aoti_torch_new_tensor_handle` | Tests creating new tensor handles that share memory |
| `test_aoti_torch_item_bool` | Tests extracting boolean values from scalar tensors |
| `test_aoti_torch_assign_tensors_out` | Tests creating tensor views that share underlying data |

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

