/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

extern "C" {

/**
 * Creates an uninitialized tensor with specified dimensions, strides, and
 * dtyper on either CPU or CUDA device.
 *
 * @param ndim Number of dimensions in the tensor
 * @param sizes_ptr Pointer to array of dimension sizes
 * @param strides_ptr Pointer to array of strides for each dimension
 * @param dtype Data type identifier (matches PyTorch scalar types)
 * @param device_type Device type (0=CPU, 1=CUDA)
 * @param device_index Device index (must be 0 for current implementation)
 * @param ret_new_tensor Output parameter for the created tensor
 * @return AOTITorchError error code (Error::Ok on success, or an error code on
 * failure)
 */
AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor);

// Function to clear all tensors from internal storage
// TODO(gasoonjia): reuse aoti_torch_delete_tensor_object to destory tensors
void clear_all_tensors();

} // extern "C"

} // namespace cuda
} // namespace backends
} // namespace executorch
