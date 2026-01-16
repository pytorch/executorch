/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/common_shims_slim.h>

namespace executorch {
namespace backends {
namespace aoti {

// ============================================================
// Basic Property Getters - Implementations
// ============================================================

AOTITorchError aoti_torch_get_data_ptr(Tensor* tensor, void** ret_data_ptr) {
  if (tensor == nullptr || ret_data_ptr == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_data_ptr = tensor->data_ptr();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_sizes(Tensor* tensor, int64_t** ret_sizes) {
  if (tensor == nullptr || ret_sizes == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_sizes = const_cast<int64_t*>(tensor->sizes().data());
  return Error::Ok;
}

AOTITorchError aoti_torch_get_strides(Tensor* tensor, int64_t** ret_strides) {
  if (tensor == nullptr || ret_strides == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_strides = const_cast<int64_t*>(tensor->strides().data());
  return Error::Ok;
}

AOTITorchError aoti_torch_get_dtype(Tensor* tensor, int32_t* ret_dtype) {
  if (tensor == nullptr || ret_dtype == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_dtype = static_cast<int32_t>(tensor->dtype());
  return Error::Ok;
}

AOTITorchError aoti_torch_get_dim(Tensor* tensor, int64_t* ret_dim) {
  if (tensor == nullptr || ret_dim == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_dim = static_cast<int64_t>(tensor->dim());
  return Error::Ok;
}

// ============================================================
// Storage & Device Property Getters - Implementations
// ============================================================

AOTITorchError aoti_torch_get_storage_offset(
    Tensor* tensor,
    int64_t* ret_storage_offset) {
  if (tensor == nullptr || ret_storage_offset == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_storage_offset = tensor->storage_offset();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_size(Tensor* tensor, int64_t* ret_size) {
  if (tensor == nullptr || ret_size == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_size = static_cast<int64_t>(tensor->storage()->nbytes());
  return Error::Ok;
}

AOTITorchError aoti_torch_get_device_type(
    Tensor* tensor,
    int32_t* ret_device_type) {
  if (tensor == nullptr || ret_device_type == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_device_type = static_cast<int32_t>(tensor->device_type());
  return Error::Ok;
}

AOTITorchError aoti_torch_get_device_index(
    Tensor* tensor,
    int32_t* ret_device_index) {
  if (tensor == nullptr || ret_device_index == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_device_index = static_cast<int32_t>(tensor->device_index());
  return Error::Ok;
}

} // namespace aoti
} // namespace backends
} // namespace executorch
