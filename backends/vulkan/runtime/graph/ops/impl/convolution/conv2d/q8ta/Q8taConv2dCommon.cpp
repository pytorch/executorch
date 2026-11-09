/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dCommon.h>

namespace vkcompute {

ActivationType activation_type_from_string(const std::string& activation) {
  if (activation == "none") {
    return ActivationType::kNone;
  } else if (activation == "relu") {
    return ActivationType::kRelu;
  }
  VK_THROW("Unknown activation type: ", activation);
}

bool q8ta_conv2d_check_packed_dim_info(const api::PackedDimInfo& info) {
  return info.packed_dim == WHCN::kChannelsDim &&
      info.packed_dim_block_size == 4 &&
      info.outer_packed_dim == WHCN::kWidthDim &&
      (info.outer_packed_dim_block_size == 1 ||
       info.outer_packed_dim_block_size == 4);
}

bool q8ta_conv2d_check_4w4c_packed_dim_info(const api::PackedDimInfo& info) {
  return info.packed_dim == WHCN::kChannelsDim &&
      info.packed_dim_block_size == 4 &&
      info.outer_packed_dim == WHCN::kWidthDim &&
      info.outer_packed_dim_block_size == 4;
}

} // namespace vkcompute
