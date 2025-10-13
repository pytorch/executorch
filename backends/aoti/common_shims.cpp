/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/runtime/platform/log.h>
#include <cstdint>

namespace executorch {
namespace backends {
namespace aoti {

namespace internal {
// Global storage for tensor metadata
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_sizes;
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_strides;
} // namespace internal

extern "C" {

// Autograd mode functions
int32_t aoti_torch_grad_mode_is_enabled() {
  // No autograd ever
  return false;
}

void aoti_torch_grad_mode_set_enabled(bool enabled) {
  if (enabled) {
    throw std::runtime_error("Cannot enable autograd");
  }
}

// Tensor attribute operations
AOTITorchError aoti_torch_get_data_ptr(Tensor* tensor, void** ret_data_ptr) {
  *ret_data_ptr = tensor->mutable_data_ptr();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_offset(
    Tensor* tensor,
    int64_t* ret_storage_offset) {
  // Storage offset is always 0 in ET
  *ret_storage_offset = 0;

  return Error::Ok;
}

AOTITorchError aoti_torch_get_strides(Tensor* tensor, int64_t** ret_strides) {
  auto it = internal::tensor_to_strides.find(tensor);
  bool needs_update = false;

  if (it == internal::tensor_to_strides.end()) {
    needs_update = true;
  } else {
    // CRITICAL: Multimodal models reuse tensors with different shapes across
    // executions (e.g., variable-length audio). We MUST validate cached
    // metadata matches current tensor state, or CUDA kernels will receive
    // incorrect shapes leading to memory corruption and segfaults.
    auto tensor_strides = tensor->strides();
    needs_update = !std::equal(
        it->second.begin(),
        it->second.end(),
        tensor_strides.begin(),
        tensor_strides.end());
  }

  if (needs_update) {
    std::vector<int64_t> strides(tensor->dim());
    auto tensor_strides = tensor->strides();
    for (int i = 0; i < tensor->dim(); i++) {
      strides[i] = tensor_strides[i];
    }
    it =
        internal::tensor_to_strides.insert_or_assign(tensor, std::move(strides))
            .first;
  }

  // For 0D tensors, data() returns nullptr on empty vectors, but we need to
  // return a valid pointer
  if (it->second.empty()) {
    static int64_t empty_strides_placeholder = 0;
    *ret_strides = &empty_strides_placeholder;
  } else {
    *ret_strides = it->second.data();
  }

  return Error::Ok;
}

AOTITorchError aoti_torch_get_dtype(Tensor* tensor, int32_t* ret_dtype) {
  *ret_dtype = static_cast<int32_t>(tensor->scalar_type());

  return Error::Ok;
}

AOTITorchError aoti_torch_get_sizes(Tensor* tensor, int64_t** ret_sizes) {
  auto it = internal::tensor_to_sizes.find(tensor);
  bool needs_update = false;

  if (it == internal::tensor_to_sizes.end()) {
    needs_update = true;
  } else {
    // CRITICAL: Multimodal models reuse tensors with different shapes across
    // executions (e.g., variable-length audio). We MUST validate cached
    // metadata matches current tensor state, or CUDA kernels will receive
    // incorrect shapes leading to memory corruption and segfaults.
    auto tensor_sizes = tensor->sizes();
    needs_update = !std::equal(
        it->second.begin(),
        it->second.end(),
        tensor_sizes.begin(),
        tensor_sizes.end());
  }

  if (needs_update) {
    std::vector<int64_t> sizes(tensor->dim());
    auto tensor_sizes = tensor->sizes();
    for (int i = 0; i < tensor->dim(); i++) {
      sizes[i] = tensor_sizes[i];
    }
    it = internal::tensor_to_sizes.insert_or_assign(tensor, std::move(sizes))
             .first;
  }

  // For 0D tensors, data() returns nullptr on empty vectors, but we need to
  // return a valid pointer
  if (it->second.empty()) {
    static int64_t empty_sizes_placeholder = 0;
    *ret_sizes = &empty_sizes_placeholder;
  } else {
    *ret_sizes = it->second.data();
  }

  return Error::Ok;
}

AOTITorchError aoti_torch_get_device_index(
    Tensor* tensor,
    int32_t* ret_device_index) {
  // Let's assume all tensors AOTI using are on CUDA:0
  *ret_device_index = 0;
  return Error::Ok;
}

AOTITorchError aoti_torch_get_dim(Tensor* tensor, int64_t* ret_dim) {
  *ret_dim = static_cast<int64_t>(tensor->dim());
  return Error::Ok;
}

// Device and layout utility functions
int32_t aoti_torch_device_type_cpu() {
  // Let's say cpu is 0 for ET as well
  return 0;
}

int32_t aoti_torch_layout_strided() {
  // ET only support strided layout, the return value will always be 0, a.k.a
  // at::Layout::Strided;
  return 0;
}

// Dtype constants - these return the PyTorch dtype codes
int32_t aoti_torch_dtype_float32() {
  return 6; // PyTorch's float32 dtype code
}

int32_t aoti_torch_dtype_bfloat16() {
  return 15; // PyTorch's bfloat16 dtype code
}

int32_t aoti_torch_dtype_int64() {
  return 4; // PyTorch's int64 dtype code
}

// Dtype utility function needed by Metal backend.
// Returns the size of the dtype in bytes.
size_t aoti_torch_dtype_element_size(int32_t dtype) {
  return dtype_to_element_size(dtype);
}

// Cleanup functions
void cleanup_tensor_metadata() {
  internal::tensor_to_sizes.clear();
  internal::tensor_to_strides.clear();
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
