// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dDirect.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/im2col/Conv2dGemm.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dRouteCommon.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dRoutes.h>

#include <algorithm>
#include <optional>

namespace vkcompute::shader_profiling {
using namespace convolution;
namespace {

std::optional<utils::StorageType> storage_option(const Json& options) {
  check_options(options, true);
  if (!options.contains("intermediate_storage")) {
    return std::nullopt;
  }
  if (!options.at("intermediate_storage").is_string()) {
    throw std::invalid_argument(
        "intermediate_storage must be a storage type name");
  }
  const std::string storage = options.at("intermediate_storage");
  if (storage == "buffer") {
    return utils::kBuffer;
  }
  if (storage == "texture2d") {
    return utils::kTexture2D;
  }
  if (storage == "texture3d") {
    return utils::kTexture3D;
  }
  throw std::invalid_argument("Unknown intermediate_storage: " + storage);
}

std::string storage_name(utils::StorageType storage) {
  switch (storage) {
    case utils::kBuffer:
      return "buffer";
    case utils::kTexture2D:
      return "texture2d";
    case utils::kTexture3D:
      return "texture3d";
  }
  throw std::invalid_argument("Unknown intermediate tensor storage");
}

bool floating(vkapi::ScalarType dtype) {
  return dtype == vkapi::kFloat || dtype == vkapi::kHalf;
}

ConvShape conv_shape(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  if (args.size() != 10u)
    throw std::invalid_argument(
        "Convolution route received wrong argument count");
  const ValueRef input = args[0];
  const ValueRef output = args.back();
  const ValueRef weight = args[1];
  ConvShape shape =
      conv_shape_base(graph, input, weight, output, graph.get_int(args[8]), 4);

  supported(
      !graph.get_bool(args[6]),
      "Explicit FP convolution routes support forward convolution only");
  supported(
      shape.input[0] == 1, "Explicit FP convolution routes require batch 1");
  supported(
      shape.weight[1] == shape.input[1] / shape.groups,
      "FP convolution weight channels disagree");
  spatial_parameter(graph, args[7], false);
  shape.kernel = {shape.weight[2], shape.weight[3]};
  check_conv_output(graph, shape, args[3], args[4], args[5]);

  const auto weight_dtype = graph.dtype_of(weight);
  const auto effective_weight_dtype = weight_dtype == vkapi::kHalf &&
          !graph.context()->adapter_ptr()->has_full_float16_buffers_support()
      ? vkapi::kFloat
      : weight_dtype;
  supported(
      floating(graph.dtype_of(input)) &&
          graph.dtype_of(input) == graph.dtype_of(output) &&
          floating(weight_dtype) &&
          effective_weight_dtype == graph.dtype_of(input),
      "FP convolution requires matching effective input/output/weight dtypes");
  supported(
      graph.storage_type_of(input) == utils::kTexture3D &&
          graph.storage_type_of(output) == utils::kTexture3D &&
          graph.packed_dim_of(input) == WHCN::kChannelsDim &&
          graph.packed_dim_of(output) == WHCN::kChannelsDim,
      "Explicit FP convolution routes require channels-packed texture3d IO");
  if (graph.val_is_not_none(args[2])) {
    constant_tensor(graph, args[2], "FP convolution bias");
    supported(
        graph.dtype_of(args[2]) == weight_dtype &&
            graph.sizes_of(args[2]) == std::vector<int64_t>{shape.output[1]},
        "FP convolution bias must match weight dtype and output channels");
  }
  return shape;
}

void check_fp_im2col_buffer_resources(
    ComputeGraph& graph,
    ValueRef input,
    const ConvShape& shape,
    std::optional<utils::StorageType> override) {
  const auto* adapter = graph.context()->adapter_ptr();
  const int64_t k = product(
      product(padded4(shape.input[1]), shape.kernel[0]), shape.kernel[1]);
  supported(k <= INT32_MAX, "FP im2col reduction size must fit int32");
  const int64_t k4 = k / 4;
  const int64_t max2d = adapter->max_texture2d_dim();
  const int64_t elem_size = vkapi::element_size(graph.dtype_of(input));
  const int64_t bytes_per_row = product(product(shape.output[3], k), elem_size);
  constexpr int64_t kScratchBudgetBytes = 16 * 1024 * 1024;
  const int64_t rows = std::min(
      shape.output[2],
      std::max(int64_t(1), kScratchBudgetBytes / bytes_per_row));
  const int64_t matrix_rows = product(rows, shape.output[3]);
  const int64_t max3d = adapter->max_texture3d_dim();
  const bool fits2d = k4 <= max2d && matrix_rows <= max2d;
  const bool fits3d = shape.output[3] <= max3d && rows <= max3d && k4 <= max3d;
  const utils::StorageType storage = override.value_or(
      graph.device_is_mali() ? utils::kBuffer
          : fits2d           ? utils::kTexture2D
          : fits3d           ? utils::kTexture3D
                             : utils::kBuffer);
  if (storage == utils::kBuffer) {
    supported(
        product(rows, bytes_per_row) <= graph.max_buffer_numel(),
        "FP im2col scratch buffer exceeds device range");
  }
}

} // namespace

Json build_conv2d_direct(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  check_options(options);
  const ConvShape shape = conv_shape(graph, args);
  grouped_packing(shape);
  conv2d_direct_impl(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args[6],
      args[7],
      args[8],
      args[9],
      false,
      0,
      0);
  return selected("direct");
}

Json build_conv2d_im2col(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  const auto storage = storage_option(options);
  const ConvShape shape = conv_shape(graph, args);
  supported(shape.groups == 1, "FP im2col convolution requires groups 1");
  check_fp_im2col_buffer_resources(graph, args[0], shape, storage);
  const size_t first = graph.execute_nodes().size();
  conv2d_gemm_impl(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args[9],
      false,
      0,
      0,
      storage);
  const ValueRef scratch =
      graph.execute_nodes().at(first)->args().at(0).refs.at(0);
  return selected(
      "im2col",
      {{"intermediate_storage", storage_name(graph.storage_type_of(scratch))}});
}

Json inspect_conv2d_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>&,
    size_t first,
    size_t end) {
  for (size_t i = first; i < end; ++i) {
    const auto& node = graph.execute_nodes().at(i);
    const std::string name = node->name();
    if (name.rfind("conv2d_im2col_", 0) == 0) {
      const ValueRef scratch = node->args().at(0).refs.at(0);
      return selected(
          "im2col",
          {{"intermediate_storage",
            storage_name(graph.storage_type_of(scratch))}});
    }
    if (name.rfind("conv2d_pw", 0) == 0) {
      return selected("pointwise");
    }
    if (name.rfind("conv2d_dw", 0) == 0) {
      return selected("depthwise");
    }
    if (name.rfind("conv_transpose2d", 0) == 0) {
      return selected("transposed");
    }
    if (name == "conv2d_float" || name == "conv2d_half") {
      return selected("direct");
    }
  }
  return Json();
}

REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, direct, build_conv2d_direct);
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, im2col, build_conv2d_im2col);
  REGISTER_ROUTE_INSPECTOR(aten.convolution.default, inspect_conv2d_route);
}

} // namespace vkcompute::shader_profiling
