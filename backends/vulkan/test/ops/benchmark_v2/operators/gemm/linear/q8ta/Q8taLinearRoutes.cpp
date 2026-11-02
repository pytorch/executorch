// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

namespace vkcompute::shader_profiling {
static Json q8ta_linear_gemv_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("Q8 linear gemv route accepts no options");
  VK_GET_OP_FN("et_vk.q8ta_linear_gemv.default")(graph, args);
  return {{"name", "gemv"}, {"options", Json::object()}};
}
REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_linear.default, gemv, q8ta_linear_gemv_route);
}
} // namespace vkcompute::shader_profiling
