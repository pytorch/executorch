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
    const api::StorageType storage_type) {
  switch (storage_type) {
    case api::kBuffer:
      kernel_name += "_buffer";
      break;
    case api::kTexture3D:
      kernel_name += "_texture3d";
      break;
    case api::kTexture2D:
      kernel_name += "_texture2d";
      break;
  }
}

void add_storage_type_suffix(std::string& kernel_name, const vTensor& tensor) {
  return add_storage_type_suffix(kernel_name, tensor.storage_type());
}

void add_dtype_suffix(std::string& kernel_name, const api::ScalarType dtype) {
  switch (dtype) {
    case api::kFloat:
      kernel_name += "_float";
      break;
    case api::kHalf:
      kernel_name += "_half";
      break;
    case api::kInt:
      kernel_name += "_int";
      break;
    case api::kChar:
    case api::kQInt8:
      kernel_name += "_int8";
      break;
    default:
      break;
  }
}

void add_dtype_suffix(std::string& kernel_name, const vTensor& tensor) {
  return add_dtype_suffix(kernel_name, tensor.dtype());
}

void add_ndim_suffix(std::string& kernel_name, const vTensor& tensor) {
  switch (tensor.storage_type()) {
    case api::kTexture3D:
      kernel_name += "_3d";
      break;
    case api::kTexture2D:
      kernel_name += "_2d";
      break;
    default:
      break;
  }
}

void add_memory_layout_suffix(
    std::string& kernel_name,
    api::GPUMemoryLayout layout) {
  switch (layout) {
    case api::kChannelsPacked:
      kernel_name += "_C_packed";
      break;
    case api::kHeightPacked:
      kernel_name += "_H_packed";
      break;
    case api::kWidthPacked:
      kernel_name += "_W_packed";
      break;
    default:
      break;
  }
}

void add_memory_layout_suffix(std::string& kernel_name, const vTensor& tensor) {
  return add_memory_layout_suffix(kernel_name, tensor.gpu_memory_layout());
}

} // namespace vkcompute
