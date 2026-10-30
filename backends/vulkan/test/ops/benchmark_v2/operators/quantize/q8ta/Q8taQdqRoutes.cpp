// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

namespace vkcompute::shader_profiling {
namespace {
void check_legacy_fp_layout(ComputeGraph& graph, ValueRef tensor) {
  const auto storage = graph.storage_type_of(tensor);
  const auto layout = graph.estimate_memory_layout_of(tensor);
  if (!((storage == utils::kBuffer && layout == utils::kWidthPacked) ||
        (storage == utils::kTexture3D && layout == utils::kChannelsPacked))) {
    throw UnsupportedRoute(
        "Legacy 4w4c Q/DQ requires width-packed floating buffers or "
        "channels-packed texture3d floating tensors");
  }
}
} // namespace

static Json quantize_legacy_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("Legacy quantize route accepts no options");
  const ValueRef output = args.back();
  if (graph.estimate_memory_layout_of(output) != utils::kPackedInt8_4W4C) {
    VK_GET_OP_FN("quantized_decomposed.quantize_per_tensor.default")
    (graph, args);
    return {{"name", "auto"}, {"options", Json::object()}};
  }
  check_legacy_fp_layout(graph, args[0]);
  add_quantize_and_pack_4w4c_node(graph, args[0], args[1], args[2], output);
  return {{"name", "legacy_4w4c"}, {"options", Json::object()}};
}
static Json dequantize_legacy_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("Legacy dequantize route accepts no options");
  if (graph.estimate_memory_layout_of(args[0]) != utils::kPackedInt8_4W4C) {
    VK_GET_OP_FN("quantized_decomposed.dequantize_per_tensor.default")
    (graph, args);
    return {{"name", "auto"}, {"options", Json::object()}};
  }
  check_legacy_fp_layout(graph, args.back());
  add_unpack_4w4c_and_dequantize_node(
      graph, args[0], args[1], args[2], args.back());
  return {{"name", "legacy_4w4c"}, {"options", Json::object()}};
}
REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      quantized_decomposed.quantize_per_tensor.default,
      legacy_4w4c,
      quantize_legacy_route);
  REGISTER_OPERATOR_ROUTE(
      quantized_decomposed.dequantize_per_tensor.default,
      legacy_4w4c,
      dequantize_legacy_route);
}
} // namespace vkcompute::shader_profiling
