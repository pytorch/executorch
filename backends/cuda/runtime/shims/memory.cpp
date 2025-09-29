/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/tensor_attribute.h>
#include <executorch/backends/cuda/runtime/shims/utils.h>
#include <executorch/runtime/platform/log.h>
#include <cstdint>
#include <cstdlib> // For posix_memalign
#include <memory>
#include <unordered_set>
#include <vector>

// CUDA error checking macro
#define ET_CUDA_CHECK_OR_RETURN_ERROR(EXPR) \
  do {                                      \
    const cudaError_t err = EXPR;           \
    if (err == cudaSuccess) {               \
      break;                                \
    }                                       \
    ET_LOG(                                 \
        Error,                              \
        "%s:%d CUDA error: %s",             \
        __FILE__,                           \
        __LINE__,                           \
        cudaGetErrorString(err));           \
    return Error::Internal;                 \
  } while (0)

// Kernel launch check macro
#define ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR() \
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetLastError())

namespace executorch {
namespace backends {
namespace cuda {

using executorch::aten::SizesType;
using executorch::aten::StridesType;
using executorch::backends::aoti::dtype_to_element_size;
using executorch::backends::aoti::dtype_to_scalar_type;

// Global storage for tensors and their metadata
std::unordered_set<std::shared_ptr<Tensor>> tensors;

extern "C" {

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor) {
  // Check that device_index is always 0
  if (device_index != 0) {
    ET_LOG(Error, "device_index must be 0, got: %d", device_index);
    return Error::InvalidArgument;
  }

  // This requires us to reserve CUDA memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }

  AOTITorchError dtype_error = validate_dtype(dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  size_t element_size = dtype_to_element_size(dtype);
  if (element_size == 0) {
    ET_LOG(Error, "Invalid element size for dtype: %d", dtype);
    return Error::InvalidArgument;
  }
  int64_t nbytes = numel * element_size;

  if (device_type == 1) { // cuda
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMallocManaged(&ptr, nbytes));
  } else if (device_type == 0) { // cpu
    // Ensure 16-byte alignment for CPU memory to match CUDA requirements
    int result = posix_memalign(&ptr, 16, nbytes);
    if (result != 0) {
      ET_LOG(Error, "Failed to allocate aligned CPU memory");
      return Error::MemoryAllocationFailed;
    }
    if (ptr == nullptr) {
      ET_LOG(Error, "Failed to call posix_memalign");
      return Error::MemoryAllocationFailed;
    }
  } else {
    ET_LOG(
        Error,
        "Need to implement empty_strided for non-CUDA non-CPU device type %d",
        device_type);
    return Error::NotImplemented;
  }

  // ETensor sizes
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // ETensor strides
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // ETensor creation with dynamic shape support for edge cases
  auto tensor = executorch::extension::from_blob(
      ptr, sizes, strides, dtype_to_scalar_type(dtype));

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);
  *ret_new_tensor = tensor.get();

  return Error::Ok;
}

// TODO(gasoonjia): reuse aoti_torch_delete_tensor_object to destory tensors
void clear_all_tensors() {
  tensors.clear();
}

} // extern "C"

} // namespace cuda
} // namespace backends
} // namespace executorch
