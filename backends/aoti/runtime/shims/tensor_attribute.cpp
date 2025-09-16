/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tensor_attribute.h"
#include <iostream>
#include "utils.h"

namespace executorch {
namespace backends {
namespace aoti {

// Global storage for tensor metadata
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_sizes;
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_strides;

extern "C" {

int32_t aoti_torch_grad_mode_is_enabled() {
  // No autograd ever
  return false;
}

void aoti_torch_grad_mode_set_enabled(bool enabled) {
  if (enabled) {
    throw std::runtime_error("Cannot enable autograd");
  }
}

AOTITorchError aoti_torch_get_data_ptr(
    AOTITensorHandle tensor,
    void** ret_data_ptr) {
  *ret_data_ptr = tensor->mutable_data_ptr();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_offset(
    AOTITensorHandle tensor,
    int64_t* ret_storage_offset) {
  // Storage offset is always 0 in ET
  *ret_storage_offset = 0;

  return Error::Ok;
}

AOTITorchError aoti_torch_get_strides(
    AOTITensorHandle tensor,
    int64_t** ret_strides) {
  auto it = tensor_to_strides.find(tensor);
  if (it == tensor_to_strides.end()) {
    std::vector<int64_t> strides(tensor->dim());
    auto tensor_strides = tensor->strides();
    for (int i = 0; i < tensor->dim(); i++) {
      strides[i] = tensor_strides[i];
    }
    it = tensor_to_strides.emplace(tensor, std::move(strides)).first;
  }
  *ret_strides = it->second.data();

  return Error::Ok;
}

AOTITorchError aoti_torch_get_dtype(
    AOTITensorHandle tensor,
    int32_t* ret_dtype) {
  *ret_dtype = static_cast<int32_t>(tensor->scalar_type());

  // ASSERTION: Only float32 tensors are supported
  AOTITorchError dtype_error = validate_dtype(*ret_dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  return Error::Ok;
}

AOTITorchError aoti_torch_get_sizes(
    AOTITensorHandle tensor,
    int64_t** ret_sizes) {
  auto it = tensor_to_sizes.find(tensor);
  if (it == tensor_to_sizes.end()) {
    std::vector<int64_t> sizes(tensor->dim());
    auto tensor_sizes = tensor->sizes();
    for (int i = 0; i < tensor->dim(); i++) {
      sizes[i] = tensor_sizes[i];
    }
    it = tensor_to_sizes.emplace(tensor, std::move(sizes)).first;
  }
  *ret_sizes = it->second.data();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_size(
    AOTITensorHandle tensor,
    int64_t* ret_size) {
  throw std::runtime_error("Cannot get storage size on ETensor");
}

AOTITorchError aoti_torch_get_device_type(
    AOTITensorHandle tensor,
    int32_t* ret_device_type) {
  // All tensors in aoti-cuda delegate are on CUDA
  *ret_device_type = aoti_torch_device_type_cuda();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_device_index(
    AOTITensorHandle tensor,
    int32_t* ret_device_index) {
  // Let's assume all tensors AOTI using are on CUDA:0
  *ret_device_index = 0;
  return Error::Ok;
}

AOTITorchError aoti_torch_get_dim(AOTITensorHandle tensor, int64_t* ret_dim) {
  *ret_dim = static_cast<int64_t>(tensor->dim());
  return Error::Ok;
}

int32_t aoti_torch_device_type_cpu() {
  // Let's say cpu is 0 for ET as well
  return 0;
}

__attribute__((__visibility__("default"))) int32_t aoti_torch_layout_strided() {
  // ET only support strided layout, the return value will always be 0, a.k.a
  // at::Layout::Strided;
  return 0;
}

__attribute__((__visibility__("default"))) int32_t
aoti_torch_device_type_cuda() {
  // Let's say cuda is 1 for ET as well
  return 1;
}

// Dtype constants - these return the PyTorch dtype codes
// Currently only float32 is supported, but using robust enum-based approach
__attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_float32() {
  return static_cast<int32_t>(SupportedDTypes::FLOAT32);
}

// Future dtype support (commented out for now):
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_bool() {
//   return static_cast<int32_t>(SupportedDTypes::BOOL);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_uint8() {
//   return static_cast<int32_t>(SupportedDTypes::UINT8);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_int8() {
//   return static_cast<int32_t>(SupportedDTypes::INT8);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_int16() {
//   return static_cast<int32_t>(SupportedDTypes::INT16);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_int32() {
//   return static_cast<int32_t>(SupportedDTypes::INT32);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_int64() {
//   return static_cast<int32_t>(SupportedDTypes::INT64);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_float16() {
//   return static_cast<int32_t>(SupportedDTypes::FLOAT16);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_float64() {
//   return static_cast<int32_t>(SupportedDTypes::FLOAT64);
// }
// 
// __attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_bfloat16() {
//   return static_cast<int32_t>(SupportedDTypes::BFLOAT16);
// }

void cleanup_tensor_metadata() {
  tensor_to_sizes.clear();
  tensor_to_strides.clear();
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
