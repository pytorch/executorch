/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

struct Conv2DParams {
  utils::ivec2 kernel_size;
  utils::ivec2 stride;
  utils::ivec2 padding;
  utils::ivec2 dilation;
  int32_t groups;
  int32_t out_channels_per_group;
  int32_t in_channels_per_group;
  int32_t logical_K_per_group;
  int32_t K_per_group;
  int32_t K4_per_group;
  int32_t logical_K;
  int32_t K;
  int32_t K4;
};

Conv2DParams create_conv2d_params(
    ComputeGraph& graph,
    const ValueRef& conv_input,
    const ValueRef& conv_output,
    const ValueRef& kernel_size,
    const ValueRef& stride,
    const ValueRef& padding,
    const ValueRef& dilation,
    const ValueRef& groups);

vkapi::SpecVarList GenerateSpecConstants(
    ComputeGraph& graph,
    Conv2DParams& conv_params,
    const ValueRef& groups,
    uint32_t apply_bias = 1);

} // namespace vkcompute
