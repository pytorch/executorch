/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/types.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// Global storage declarations
extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
// A live handle, and the delegate it was created for (see
// metal_begin_delegate), or nullptr if none was current.
struct TrackedTensor {
  std::shared_ptr<Tensor> tensor;
  const void* delegate;
};
// Maps raw Tensor* → its TrackedTensor for O(1) lookup/deletion.
extern std::unordered_map<Tensor*, TrackedTensor> tensors;

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

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle);

void cleanup_memory();

// Shim state is shared by every delegate in the process, so each one records
// what it creates. metal_begin_delegate registers a delegate. While a
// MetalDelegateScope for it is alive on a thread, the handles created there
// are recorded against it, and while it is loading so are the buffers
// aoti_torch_mps_malloc makes (its constants). metal_release_delegate frees
// what was recorded for it and returns whether no delegate is left, when the
// caller can reset everything with cleanup_memory().
void metal_begin_delegate(const void* delegate);
bool metal_release_delegate(const void* delegate);
// For aoti_torch_mps_malloc, and for whatever frees or recycles a buffer.
void metal_note_delegate_buffer(void* ptr);
void metal_forget_delegate_buffer(void* ptr);

// The memory at [ptr, ptr + nbytes) is freed (aoti_torch_mps_free) while
// handles in it may live on. They keep nothing there any more: deleting one
// later only drops the handle, so it cannot unregister or uncount what a later
// allocation puts at its address. No handle may own memory there.
void retire_handles_within(void* ptr, size_t nbytes);

} // extern "C"

class MetalDelegateScope {
 public:
  MetalDelegateScope(const void* delegate, bool loading);
  ~MetalDelegateScope();
  MetalDelegateScope(const MetalDelegateScope&) = delete;
  MetalDelegateScope& operator=(const MetalDelegateScope&) = delete;

 private:
  const void* previous_;
  bool previous_loading_;
};

} // namespace metal
} // namespace backends
} // namespace executorch
