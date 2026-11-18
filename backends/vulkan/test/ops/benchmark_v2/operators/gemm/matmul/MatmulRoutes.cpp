// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/GemmCommon.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/Linear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/LinearCoopmat.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/matmul/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/matmul/MatmulCoopmat.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
namespace vkcompute::shader_profiling {
namespace {
template <bool Coop, bool Linear>
Json build_gemm_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("GEMM routes accept no options");
  const ValueRef input = args.at(0), weight = args.at(1), out = args.back();
  if constexpr (Coop) {
    const auto* adapter = graph.context()->adapter_ptr();
    const auto sizes = graph.sizes_of(input), outs = graph.sizes_of(out);
    if (!adapter->supports_cooperative_matrix() ||
        adapter->subgroup_size() != 64 || !graph.is_buffer_storage(out) ||
        sizes.size() != 2 || sizes[0] % 64 || sizes[1] % 32 || outs.back() % 64)
      throw UnsupportedRoute(
          "coopmat requires cooperative matrices, subgroup64, buffer output, and M/N/K alignment64/64/32");
  }
  if (Linear || graph.val_is_tref(weight)) {
    const int64_t batch =
        graph.sizes_of(weight).size() == 3 ? graph.sizes_of(weight).at(0) : 1;
    const ValueRef packed =
        prepack_fp_linear_weight(graph, weight, Linear, batch, Coop);
    const bool has_bias = Linear && graph.val_is_not_none(args.at(2));
    ValueRef bias = kDummyValueRef;
    if (has_bias)
      bias = prepack_standard(
          graph,
          args.at(2),
          graph.storage_type_of(out),
          utils::kWidthPacked,
          false);
    if constexpr (Coop)
      add_linear_coopmat_node(
          graph,
          input,
          packed,
          bias,
          has_bias,
          out,
          utils::safe_downcast<int32_t>(batch));
    else
      add_linear_tiled_node(
          graph,
          input,
          packed,
          bias,
          has_bias,
          out,
          utils::safe_downcast<int32_t>(batch));
  } else {
    if constexpr (Coop)
      add_matmul_coopmat_node(graph, input, weight, out);
    else
      add_matmul_tiled_node(graph, input, weight, out);
  }
  return {{"name", Coop ? "coopmat" : "tiled"}, {"options", options}};
}
REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      aten.mm.default, tiled, (build_gemm_route<false, false>));
  REGISTER_OPERATOR_ROUTE(
      aten.mm.default, coopmat, (build_gemm_route<true, false>));
  REGISTER_OPERATOR_ROUTE(
      aten.linear.default, tiled, (build_gemm_route<false, true>));
  REGISTER_OPERATOR_ROUTE(
      aten.linear.default, coopmat, (build_gemm_route<true, true>));
}
} // namespace
} // namespace vkcompute::shader_profiling
