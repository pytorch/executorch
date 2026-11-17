// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/SDPA.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

namespace vkcompute::shader_profiling {
namespace {

template <int64_t Override>
Json build_sdpa_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("SDPA routes accept no options");
  if (args.size() != 9)
    throw std::invalid_argument("SDPA expects nine production arguments");
  sdpa_impl_with_shader_override(graph, args, graph.add_scalar(Override));
  const char* name = Override == kShaderOverrideForceGqa ? "gqa"
      : Override == kShaderOverrideForceTile2            ? "gqa_tile2"
      : Override == kShaderOverrideForceBase             ? "gqa_base"
                                                         : "non_gqa";
  return {{"name", name}, {"options", options}};
}

REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      llama.custom_sdpa.default,
      gqa,
      (build_sdpa_route<kShaderOverrideForceGqa>));
  REGISTER_OPERATOR_ROUTE(
      llama.custom_sdpa.default,
      non_gqa,
      (build_sdpa_route<kShaderOverrideForceNonGqa>));
  REGISTER_OPERATOR_ROUTE(
      llama.custom_sdpa.default,
      gqa_tile2,
      (build_sdpa_route<kShaderOverrideForceTile2>));
  REGISTER_OPERATOR_ROUTE(
      llama.custom_sdpa.default,
      gqa_base,
      (build_sdpa_route<kShaderOverrideForceBase>));
}

} // namespace
} // namespace vkcompute::shader_profiling
