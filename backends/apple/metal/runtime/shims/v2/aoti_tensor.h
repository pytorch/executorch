/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI tensor + memory layer for the v2 Metal backend.
//
// One-stop shop for everything tensor-and-memory the AOTI .so calls into:
//   - Tensor lifecycle: create_tensor_from_blob_v2 / empty_strided / delete /
//     copy_ / _reinterpret_tensor / new_tensor_handle / cleanup_memory
//   - MPS buffer shims: mps_malloc / mps_free / mps_memcpy / mps_copy_buffer
//   - MPS device-type override: aoti_torch_get_device_type / device_type_mps
//
// All operate on SlimTensor handles (see aoti_types.h). Implementations
// route through the metal_* C ABI in runtime.h.
//
// NOTE: These symbols intentionally collide with v1's. v1 and v2 must
// live in separate static libraries; users link exactly one.

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// =====================================================================
// Global storage (definitions in aoti_tensor.cpp)
// =====================================================================

extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
// Maps raw SlimTensor* -> unique_ptr<SlimTensor> for O(1) lookup/deletion.
extern std::unordered_map<Tensor*, std::unique_ptr<Tensor>> tensors;

// =====================================================================
// Tensor lifecycle
// =====================================================================

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

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle);

void cleanup_memory();

// =====================================================================
// MPS buffer shims (the AOTI .so calls these directly for raw buffers)
// =====================================================================

AOTITorchError aoti_torch_mps_malloc(void** buffer, size_t num_bytes);
AOTITorchError aoti_torch_mps_free(void* ptr);

AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start);

AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset);

// =====================================================================
// MPS device-type override
// =====================================================================

// Returns the MPS device-type code (13). Stable across v1/v2.
int32_t aoti_torch_device_type_mps();

// Override common_shims_slim's default that reports the SlimTensor's
// actual device type (CPU). For v2, all tensors going through the AOTI
// shim layer are conceptually MPS regardless of how SlimTensor models
// them internally.
AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type);

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
