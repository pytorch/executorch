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
    const utils::StorageType storage_type) {
  switch (storage_type) {
    case utils::kBuffer:
      kernel_name += "_buffer";
      break;
    case utils::kTexture3D:
      kernel_name += "_texture3d";
      break;
    case utils::kTexture2D:
      kernel_name += "_texture2d";
      break;
  }
}

void add_dtype_suffix(std::string& kernel_name, const vkapi::ScalarType dtype) {
  switch (dtype) {
    case vkapi::kDouble:
      kernel_name += "_double";
      break;
    case vkapi::kFloat:
      kernel_name += "_float";
      break;
    case vkapi::kHalf:
      kernel_name += "_half";
      break;
    case vkapi::kChar:
    case vkapi::kQInt8:
      kernel_name += "_int8";
      break;
    case vkapi::kByte:
    case vkapi::kBool:
    case vkapi::kQUInt8:
      kernel_name += "_uint8";
      break;
    case vkapi::kShort:
      kernel_name += "_int16";
      break;
    case vkapi::kUInt16:
      kernel_name += "_uint16";
      break;
    case vkapi::kInt:
      kernel_name += "_int32";
      break;
    case vkapi::kUInt:
      kernel_name += "_uint32";
      break;
    case vkapi::kLong:
      kernel_name += "_int64";
      break;
    case vkapi::kUInt64:
      kernel_name += "_uint64";
      break;
    default:
      break;
  }
}

void add_packed_dim_suffix(std::string& kernel_name, const int32_t packed_dim) {
  switch (packed_dim) {
    case WHCN::kWidthDim:
      kernel_name += "_W_packed";
      break;
    case WHCN::kHeightDim:
      kernel_name += "_H_packed";
      break;
    case WHCN::kChannelsDim:
      kernel_name += "_C_packed";
      break;
    default:
      VK_THROW("Invalid packed dim!");
  }
}

} // namespace vkcompute
