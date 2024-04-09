/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void apply_dtype_suffix(std::stringstream& kernel_name, const vTensor& tensor) {
  switch (tensor.image().format()) {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      kernel_name << "_float";
      break;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      kernel_name << "_half";
      break;
    case VK_FORMAT_R32G32B32A32_SINT:
      kernel_name << "_int";
      break;
    default:
      break;
  }
}

void apply_ndim_suffix(std::stringstream& kernel_name, const vTensor& tensor) {
  switch (tensor.storage_type()) {
    case api::StorageType::TEXTURE_3D:
      kernel_name << "_3d";
      break;
    case api::StorageType::TEXTURE_2D:
      kernel_name << "_2d";
      break;
    default:
      break;
  }
}

void apply_memory_layout_suffix(
    std::stringstream& kernel_name,
    const vTensor& tensor) {
  switch (tensor.gpu_memory_layout()) {
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      kernel_name << "_C_packed";
      break;
    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      kernel_name << "_H_packed";
      break;
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      kernel_name << "_W_packed";
      break;
    default:
      break;
  }
}

} // namespace vkcompute
