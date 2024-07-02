/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_storage_type_suffix(
    std::string& kernel_name,
    const vkapi::StorageType storage_type) {
  switch (storage_type) {
    case vkapi::kBuffer:
      kernel_name += "_buffer";
      break;
    case vkapi::kTexture3D:
      kernel_name += "_texture3d";
      break;
    case vkapi::kTexture2D:
      kernel_name += "_texture2d";
      break;
  }
}

void add_storage_type_suffix(
    std::string& kernel_name,
    const api::vTensor& tensor) {
  return add_storage_type_suffix(kernel_name, tensor.storage_type());
}

void add_dtype_suffix(std::string& kernel_name, const vkapi::ScalarType dtype) {
  switch (dtype) {
    case vkapi::kFloat:
      kernel_name += "_float";
      break;
    case vkapi::kHalf:
      kernel_name += "_half";
      break;
    case vkapi::kInt:
      kernel_name += "_int";
      break;
    case vkapi::kChar:
    case vkapi::kQInt8:
      kernel_name += "_int8";
      break;
    default:
      break;
  }
}

void add_dtype_suffix(std::string& kernel_name, const api::vTensor& tensor) {
  return add_dtype_suffix(kernel_name, tensor.dtype());
}

void add_ndim_suffix(std::string& kernel_name, const api::vTensor& tensor) {
  switch (tensor.storage_type()) {
    case vkapi::kTexture3D:
      kernel_name += "_3d";
      break;
    case vkapi::kTexture2D:
      kernel_name += "_2d";
      break;
    default:
      break;
  }
}

void add_memory_layout_suffix(
    std::string& kernel_name,
    vkapi::GPUMemoryLayout layout) {
  switch (layout) {
    case vkapi::kChannelsPacked:
      kernel_name += "_C_packed";
      break;
    case vkapi::kHeightPacked:
      kernel_name += "_H_packed";
      break;
    case vkapi::kWidthPacked:
      kernel_name += "_W_packed";
      break;
    default:
      break;
  }
}

void add_memory_layout_suffix(
    std::string& kernel_name,
    const api::vTensor& tensor) {
  return add_memory_layout_suffix(kernel_name, tensor.gpu_memory_layout());
}

} // namespace vkcompute
