/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tensor + memory C ABI the AOTI .so calls into.
//
//   - Tensor lifecycle: create_tensor_from_blob_v2 / empty_strided / delete /
//     copy_ / _reinterpret_tensor / new_tensor_handle / cleanup_memory
//   - MPS buffer shims:  mps_malloc / mps_free / mps_memcpy / mps_copy_buffer
//   - Device type:       aoti_torch_get_device_type / device_type_mps
//
// All operate on SlimTensor handles (see aoti_types.h). Buffer ops route
// through the metal_* C ABI in runtime.h.
//
// Thread safety: every public entry point that mutates `tensors` or
// `memory_to_n_tensor` (declared extern below) acquires `tensors_mutex`
// for its entire critical section. External writers of these maps must
// do the same. The mutex is non-recursive — public APIs must not call
// each other while holding the lock.
//
// These symbols intentionally collide with the v1 backend; link exactly
// one of v1/v2 into a process.

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// Globals (defined in aoti_tensor.cpp). Guarded by tensors_mutex.
extern std::mutex tensors_mutex;
extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
extern std::unordered_map<Tensor*, std::unique_ptr<Tensor>> tensors;

// Tensor lifecycle.

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

// Register an existing MPS-allocated buffer as a fresh OWNING tensor
// (refcount = 1) atomically. Use this in preference to the
// create_tensor_from_blob_v2 → memory_to_n_tensor[ptr] = 1 sequence,
// which exposes a window in which observers see the tensor as
// externally-owned.
AOTITorchError aoti_torch_create_owned_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    AOTITensorHandle* ret_new_tensor);

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

// Caller must guarantee no concurrent registrations. Typically called
// at backend tear-down.
void cleanup_memory();

// MPS buffer shims (called directly by the AOTI .so).

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

// Device-type override: returns MPS for all tensors flowing through the
// v2 shim, regardless of how SlimTensor models device internally.
int32_t aoti_torch_device_type_mps();

AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type);

}  // extern "C"

}  // namespace metal
}  // namespace backends
}  // namespace executorch
