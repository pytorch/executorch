// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

namespace vkcompute::shader_profiling {
namespace {
template <bool B1>
Json build_dw_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("Depthwise tile routes accept no options");
  if ((args.size() != 10 && args.size() != 17) ||
      graph.extract_scalar<bool>(args[6]) ||
      graph.extract_scalar<int64_t>(args[8]) != graph.sizes_of(args[0]).at(1))
    throw UnsupportedRoute(
        "Depthwise tile routes require forward depthwise convolution");
  if (graph.sizes_of(args[0]).at(0) != 1)
    throw UnsupportedRoute("Depthwise tile routes require batch 1");
  if (!graph.is_texture_storage(args[0]) ||
      !graph.is_texture_storage(args.back()))
    throw UnsupportedRoute("Depthwise tile routes require texture IO");
  conv2d_dw_impl(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args.back(),
      conv2d_epilogue_from_args(graph, args),
      B1);
  const auto& sizes = graph.sizes_of(args[1]);
  const auto stride = *graph.get_int_list(args[3]);
  const auto dilation = *graph.get_int_list(args[5]);
  const bool tiled = sizes[2] == sizes[3] && (sizes[2] == 3 || sizes[2] == 5);
  const bool selected = B1 && tiled && stride == dilation;
  return {{"name", selected ? "b1x1" : "b4x2"}, {"options", options}};
}
REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, b1x1, (build_dw_route<true>));
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, b4x2, (build_dw_route<false>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.conv2d.with_epilogue, b1x1, (build_dw_route<true>));
  REGISTER_OPERATOR_ROUTE(
      et_vk.conv2d.with_epilogue, b4x2, (build_dw_route<false>));
}
} // namespace
} // namespace vkcompute::shader_profiling
