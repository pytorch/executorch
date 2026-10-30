/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <string>

namespace vkcompute {

enum class ActivationType : uint32_t {
  kNone = 0,
  kRelu = 1,
};

ActivationType activation_type_from_string(const std::string& activation);

bool q8ta_conv2d_check_packed_dim_info(const api::PackedDimInfo& info);

bool q8ta_conv2d_check_4w4c_packed_dim_info(const api::PackedDimInfo& info);

} // namespace vkcompute
