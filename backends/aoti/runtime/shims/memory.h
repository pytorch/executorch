/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef SUPPORT_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "types.h"

namespace executorch {
namespace backends {
namespace aoti {

extern "C" {

// Global storage declarations
extern std::unordered_map<Tensor*, bool> is_tensor_own_memory;
extern std::unordered_set<std::shared_ptr<Tensor>> tensors;

// Memory-related operations
AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor);

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor);

AOTITorchError aoti_torch_copy_(
    AOTITensorHandle self,
    AOTITensorHandle src,
    int32_t non_blocking);

AOTITorchError aoti_torch__reinterpret_tensor(
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AOTITensorHandle* ret_new_tensor);

// Utility functions
#ifdef SUPPORT_CUDA
AOTITorchError checkCudaError(cudaError_t err, const char* msg);
#endif
void cleanup_memory();

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
