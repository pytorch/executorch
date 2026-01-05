/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/export.h>
#include <executorch/runtime/core/error.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Uses conditional compilation to separate the implementation between
// CUDA backend (SlimTensor) and other backends like MPS (ETensor).
// The caller determines which path is used by defining CUDA_AVAILABLE.
#ifdef CUDA_AVAILABLE
#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#else
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#endif

namespace executorch {
namespace backends {
namespace aoti {

// Common using declarations for ExecuTorch types
using executorch::runtime::Error;

// ============================================================
// Tensor Type Definition - branched based on CUDA_AVAILABLE
// ============================================================
#ifdef CUDA_AVAILABLE
using Tensor = executorch::backends::aoti::slim::SlimTensor;
#else
using Tensor = executorch::runtime::etensor::Tensor;
#endif

// Common AOTI type aliases
using AOTIRuntimeError = Error;
using AOTITorchError = Error;

#ifndef CUDA_AVAILABLE
namespace internal {
// Global storage for tensor metadata (ETensor path only)
// SlimTensor stores sizes/strides directly in int64_t[] - no caching needed
inline std::unordered_map<Tensor*, std::vector<int64_t>>& tensor_to_sizes() {
  static std::unordered_map<Tensor*, std::vector<int64_t>> instance;
  return instance;
}
inline std::unordered_map<Tensor*, std::vector<int64_t>>& tensor_to_strides() {
  static std::unordered_map<Tensor*, std::vector<int64_t>> instance;
  return instance;
}
} // namespace internal
#endif

// ============================================================
// Basic Property Getters - Inline implementations
// ============================================================

inline AOTITorchError aoti_torch_get_data_ptr(
    Tensor* tensor,
    void** ret_data_ptr) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_data_ptr == nullptr) {
    return Error::InvalidArgument;
  }

#ifdef CUDA_AVAILABLE
  *ret_data_ptr = tensor->data_ptr();
#else
  *ret_data_ptr = tensor->mutable_data_ptr();
#endif
  return Error::Ok;
}

inline AOTITorchError aoti_torch_get_sizes(
    Tensor* tensor,
    int64_t** ret_sizes) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_sizes == nullptr) {
    return Error::InvalidArgument;
  }

#ifdef CUDA_AVAILABLE
  // SlimTensor stores sizes directly in int64_t[] - no caching needed
  *ret_sizes = const_cast<int64_t*>(tensor->sizes().data());
#else
  auto it = internal::tensor_to_sizes().find(tensor);
  bool needs_update = false;

  if (it == internal::tensor_to_sizes().end()) {
    needs_update = true;
  } else {
    // Validate cached metadata matches current tensor state
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
    it = internal::tensor_to_sizes()
             .insert_or_assign(tensor, std::move(sizes))
             .first;
  }

  // For 0D tensors, data() returns nullptr on empty vectors
  if (it->second.empty()) {
    static int64_t empty_sizes_placeholder = 0;
    *ret_sizes = &empty_sizes_placeholder;
  } else {
    *ret_sizes = it->second.data();
  }
#endif
  return Error::Ok;
}

inline AOTITorchError aoti_torch_get_strides(
    Tensor* tensor,
    int64_t** ret_strides) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_strides == nullptr) {
    return Error::InvalidArgument;
  }

#ifdef CUDA_AVAILABLE
  // SlimTensor stores strides directly in int64_t[] - no caching needed
  *ret_strides = const_cast<int64_t*>(tensor->strides().data());
#else
  auto it = internal::tensor_to_strides().find(tensor);
  bool needs_update = false;

  if (it == internal::tensor_to_strides().end()) {
    needs_update = true;
  } else {
    // Validate cached metadata matches current tensor state
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
    it = internal::tensor_to_strides()
             .insert_or_assign(tensor, std::move(strides))
             .first;
  }

  // For 0D tensors, data() returns nullptr on empty vectors
  if (it->second.empty()) {
    static int64_t empty_strides_placeholder = 0;
    *ret_strides = &empty_strides_placeholder;
  } else {
    *ret_strides = it->second.data();
  }
#endif
  return Error::Ok;
}

inline AOTITorchError aoti_torch_get_dtype(Tensor* tensor, int32_t* ret_dtype) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_dtype == nullptr) {
    return Error::InvalidArgument;
  }

#ifdef CUDA_AVAILABLE
  *ret_dtype = static_cast<int32_t>(tensor->dtype());
#else
  *ret_dtype = static_cast<int32_t>(tensor->scalar_type());
#endif
  return Error::Ok;
}

inline AOTITorchError aoti_torch_get_dim(Tensor* tensor, int64_t* ret_dim) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_dim == nullptr) {
    return Error::InvalidArgument;
  }

  *ret_dim = static_cast<int64_t>(tensor->dim());
  return Error::Ok;
}

// ============================================================
// Storage & Device Property Getters - Inline implementations
// ============================================================

inline AOTITorchError aoti_torch_get_storage_offset(
    Tensor* tensor,
    int64_t* ret_storage_offset) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_storage_offset == nullptr) {
    return Error::InvalidArgument;
  }

#ifdef CUDA_AVAILABLE
  // SlimTensor supports real storage offset
  *ret_storage_offset = tensor->storage_offset();
#else
  // ETensor doesn't support storage_offset, return 0
  *ret_storage_offset = 0;
#endif
  return Error::Ok;
}

inline AOTITorchError aoti_torch_get_storage_size(
    Tensor* tensor,
    int64_t* ret_size) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_size == nullptr) {
    return Error::InvalidArgument;
  }

  *ret_size = static_cast<int64_t>(tensor->nbytes());
  return Error::Ok;
}

inline AOTITorchError aoti_torch_get_device_type(
    Tensor* tensor,
    int32_t* ret_device_type) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_device_type == nullptr) {
    return Error::InvalidArgument;
  }

#ifdef CUDA_AVAILABLE
  // SlimTensor supports real device type
  *ret_device_type = static_cast<int32_t>(tensor->device_type());
#else
  // ETensor is always CPU in default mode
  *ret_device_type = 0; // CPU
#endif
  return Error::Ok;
}

inline AOTITorchError aoti_torch_get_device_index(
    Tensor* tensor,
    int32_t* ret_device_index) {
  if (tensor == nullptr) {
    return Error::InvalidArgument;
  }
  if (ret_device_index == nullptr) {
    return Error::InvalidArgument;
  }

#ifdef CUDA_AVAILABLE
  // SlimTensor supports real device index
  *ret_device_index = static_cast<int32_t>(tensor->device_index());
#else
  // ETensor doesn't support multi-device, return 0
  *ret_device_index = 0;
#endif
  return Error::Ok;
}

// ============================================================
// DType Constants - These return PyTorch ScalarType enum values
// ============================================================

inline int32_t aoti_torch_dtype_float32() {
  return 6; // ScalarType::Float
}

inline int32_t aoti_torch_dtype_bfloat16() {
  return 15; // ScalarType::BFloat16
}

inline int32_t aoti_torch_dtype_int64() {
  return 4; // ScalarType::Long
}

inline int32_t aoti_torch_dtype_int32() {
  return 3; // ScalarType::Int
}

inline int32_t aoti_torch_dtype_int16() {
  return 2; // ScalarType::Short
}

inline int32_t aoti_torch_dtype_int8() {
  return 1; // ScalarType::Char
}

inline int32_t aoti_torch_dtype_bool() {
  return 11; // ScalarType::Bool
}

// ============================================================
// Device Type Constants
// ============================================================

inline int32_t aoti_torch_device_type_cpu() {
  return 0; // DeviceType::CPU
}

inline int32_t aoti_torch_device_type_cuda() {
  return 1; // DeviceType::CUDA
}

// ============================================================
// Grad Mode Functions (not supported in ExecuTorch)
// ============================================================

inline bool aoti_torch_grad_mode_is_enabled() {
  return false; // ExecuTorch doesn't support autograd
}

inline AOTITorchError aoti_torch_grad_mode_set_enabled(bool enabled) {
  if (enabled) {
    return Error::NotSupported; // Grad mode not supported in ExecuTorch
  }
  return Error::Ok;
}

} // namespace aoti
} // namespace backends
} // namespace executorch
