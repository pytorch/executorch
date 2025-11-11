/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

namespace vkcompute {

struct Kernel1dParams final {
  int kernel_size;
  int stride;
  int padding;
  int dilation;
  int in_group_size;
  int out_group_size;
};

struct Kernel2dParams final {
  utils::ivec2 kernel_size;
  utils::ivec2 stride;
  utils::ivec2 padding;
  utils::ivec2 dilation;
};

Kernel2dParams create_kernel2d_params(
    ComputeGraph& graph,
    const ValueRef weight,
    const bool kernel_size_only,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation);

Kernel2dParams create_kernel2d_params(
    ComputeGraph& graph,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding);

int64_t calc_out_size(
    const int64_t in_size,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const bool ceil_mode);

std::vector<int64_t> calc_out_sizes_hw(
    ComputeGraph& graph,
    const std::vector<int64_t>& in_sizes,
    const ValueRef weight,
    const bool kernel_size_only,
    const std::vector<ValueRef>& args,
    const bool transposed = false);

} // namespace vkcompute
