/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace at {
namespace native {
namespace vulkan {

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const api::ShaderInfo& shader);

struct ArithmeticParams final {
  api::utils::ivec4 outputSizes;
  api::utils::ivec4 input1Sizes;
  api::utils::ivec4 input2Sizes;
  float alpha;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
