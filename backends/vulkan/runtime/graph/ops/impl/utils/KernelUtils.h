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

struct KernelParams final {
  api::utils::ivec2 kernel_size;
  api::utils::ivec2 stride;
  api::utils::ivec2 padding;
  api::utils::ivec2 dilation;
};

KernelParams create_kernel_params(
    ComputeGraph& graph,
    const ValueRef weight,
    const bool kernel_size_only,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation);

std::vector<int64_t> calc_out_sizes_hw(
    ComputeGraph& graph,
    const std::vector<int64_t>& in_sizes,
    const ValueRef weight,
    const bool kernel_size_only,
    const std::vector<ValueRef>& args,
    const bool transposed = false);

} // namespace vkcompute
