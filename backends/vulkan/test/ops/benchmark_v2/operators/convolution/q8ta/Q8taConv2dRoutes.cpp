// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dCommon.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dDirect.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dPW.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/im2col/Q8taConv2dIm2Col.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dRouteCommon.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/q8ta/Q8taConv2dRoutes.h>

#include <cmath>

namespace vkcompute::shader_profiling {
using namespace convolution;
namespace {

ConvShape conv_shape(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  if (args.size() != 16u)
    throw std::invalid_argument(
        "Convolution route received wrong argument count");
  const ValueRef input = args[0];
  const ValueRef output = args.back();
  const ValueRef weight = args[3];
  ConvShape shape =
      conv_shape_base(graph, input, weight, output, graph.get_int(args[13]), 2);

  shape.kernel = spatial_parameter(graph, args[9], true);
  const int64_t kernel_columns = product(
      product(shape.input[1] / shape.groups, shape.kernel[0]), shape.kernel[1]);
  supported(
      shape.weight[1] == padded4(kernel_columns),
      "Q8 convolution weights require aligned [Cout, Kh*Kw*Cin/groups] rows");
  check_conv_output(graph, shape, args[10], args[11], args[12]);

  supported(
      graph.dtype_of(input) == vkapi::kInt8x4 &&
          graph.dtype_of(output) == vkapi::kInt8x4 &&
          graph.is_buffer_storage(input) && graph.is_buffer_storage(output),
      "Q8 convolution requires Int8x4 buffer input/output");
  supported(
      q8ta_conv2d_check_packed_dim_info(graph.packed_dim_info_of(input)) &&
          q8ta_conv2d_check_packed_dim_info(graph.packed_dim_info_of(output)),
      "Q8 convolution requires channels-packed4 with outer width block 1 or 4");
  supported(
      graph.dtype_of(weight) == vkapi::kChar,
      "Q8 convolution weights require int8 dtype");
  for (size_t i : {4, 5, 8}) {
    if (i == 8 && graph.val_is_none(args[i])) {
      continue;
    }
    constant_tensor(graph, args[i], "Q8 convolution channel parameters");
    supported(
        graph.dtype_of(args[i]) == (i == 4 ? vkapi::kInt : vkapi::kFloat) &&
            graph.dim_of(args[i]) == 1 &&
            graph.size_at<int64_t>(0, args[i]) >= shape.output[1],
        "Q8 convolution sums/scales/bias require canonical int32/float32 channel vectors");
  }
  for (size_t i : {1, 6}) {
    const double scale = graph.extract_scalar<double>(args[i]);
    supported(
        std::isfinite(scale) && scale > 0,
        "Q8 convolution scales must be positive and finite");
  }
  for (size_t i : {2, 7}) {
    const int64_t zero_point = graph.extract_scalar<int64_t>(args[i]);
    supported(
        zero_point >= -128 && zero_point <= 127,
        "Q8 convolution zero points must fit int8");
  }
  const std::string activation = graph.extract_string(args[14]);
  supported(
      activation == "none" || activation == "relu",
      "Q8 convolution supports none/relu activation");
  return shape;
}

} // namespace

Json build_q8ta_conv2d_direct(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  check_options(options);
  const ConvShape shape = conv_shape(graph, args);
  grouped_packing(shape);
  q8ta_conv2d_general_impl(graph, args);
  return selected("direct");
}

namespace {
enum class DotRoute { Auto, Signed, Unsigned, Fallback };

Json build_im2col(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options,
    DotRoute dot,
    const char* route_name) {
  check_options(options);
  const ConvShape shape = conv_shape(graph, args);
  grouped_packing(shape);
  supported(
      shape.input[1] / shape.groups % 4 == 0,
      "Q8 im2col convolution requires input channels per group divisible by4");
  const auto sizes =
      calculate_q8ta_im2col_sizes(&graph, args[0], args[15], args[9], args[13]);
  const auto plan = make_q8ta_conv2d_stream_plan_for_device(
      sizes[0], sizes[1], sizes[2], sizes[3], graph.max_buffer_numel());
  supported(plan.feasible, "Q8 im2col has no feasible device scratch plan");
  if (dot == DotRoute::Unsigned) {
    supported(
        graph.can_use_int8_dot_product() &&
            shape.weight[1] <= kMaxUnsignedDotAccumulatorBytes,
        "Unsigned Q8 im2col requires packed integer dot and bounded K");
    q8ta_conv2d_im2col_impl(graph, true, args);
  } else if (dot == DotRoute::Signed) {
    q8ta_conv2d_im2col_impl(graph, false, args);
  } else if (dot == DotRoute::Fallback) {
    vkapi::ScopedAdapterCapabilityOverride no_dot(
        graph.context()->adapter_ptr(),
        vkapi::AdapterCapabilityOverrides::without_dot_product_support());
    q8ta_conv2d_im2col_impl(graph, false, args);
  } else {
    q8ta_conv2d_im2col(graph, args);
  }
  return selected(route_name);
}

Json build_im2col_signed(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  return build_im2col(graph, args, options, DotRoute::Signed, "im2col");
}
Json build_im2col_unsigned(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  return build_im2col(
      graph, args, options, DotRoute::Unsigned, "im2col_unsigned");
}
Json build_im2col_fallback(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  return build_im2col(
      graph, args, options, DotRoute::Fallback, "im2col_fallback");
}

Json build_pointwise(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options,
    DotRoute dot,
    const char* route_name) {
  check_options(options);
  const ConvShape shape = conv_shape(graph, args);
  supported(
      shape.kernel == std::array<int64_t, 2>{1, 1} && shape.groups == 1 &&
          spatial_parameter(graph, args[10], true) ==
              std::array<int64_t, 2>{1, 1} &&
          spatial_parameter(graph, args[11], false) ==
              std::array<int64_t, 2>{0, 0},
      "Q8 pointwise route requires ungrouped1x1, unit stride, zero padding");
  if (dot == DotRoute::Unsigned) {
    supported(
        graph.can_use_int8_dot_product() &&
            shape.weight[1] <= kMaxUnsignedDotAccumulatorBytes,
        "Unsigned Q8 pointwise requires packed integer dot and bounded K");
    q8ta_conv2d_pw_impl(graph, true, args);
  } else if (dot == DotRoute::Signed) {
    q8ta_conv2d_pw_impl(graph, false, args);
  } else {
    q8ta_conv2d_pw(graph, args);
  }
  return selected(route_name);
}
Json build_pw_signed(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  return build_pointwise(graph, args, options, DotRoute::Signed, "pw_signed");
}
Json build_pw_unsigned(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  return build_pointwise(
      graph, args, options, DotRoute::Unsigned, "pw_unsigned");
}
Json build_pw_auto(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  return build_pointwise(graph, args, options, DotRoute::Auto, "pw_auto");
}

Json build_legacy_4w4c(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  check_options(options);
  supported(args.size() == 16, "Q8 legacy route requires16 native arguments");
  for (ValueRef tensor : {args[0], args[15]}) {
    const auto& packing = graph.packed_dim_info_of(tensor);
    supported(
        graph.dtype_of(tensor) == vkapi::kInt8x4 &&
            graph.is_buffer_storage(tensor) &&
            q8ta_conv2d_check_packed_dim_info(packing) &&
            packing.outer_packed_dim_block_size == 4,
        "Q8 legacy route requires4w4c input/output buffers");
  }
  supported(
      graph.extract_string(args[14]) == "none",
      "Q8 legacy route has no fused activation argument");
  std::vector<ValueRef> native(args.begin(), args.begin() + 14);
  native.push_back(args[15]);
  VK_GET_OP_FN("et_vk.conv2d_q8ta_q8csw_q8to.default")(graph, native);
  return selected("legacy_4w4c");
}
} // namespace

Json build_q8ta_conv2d_im2col(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  return build_im2col(graph, args, options, DotRoute::Auto, "im2col");
}

Json inspect_q8ta_conv2d_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>&,
    size_t first,
    size_t end) {
  for (size_t i = first; i < end; ++i) {
    const std::string name = graph.execute_nodes().at(i)->name();
    if (name == "q8ta_im2col") {
      return selected("im2col");
    }
    if (name.rfind("q8ta_conv2d_pw", 0) == 0) {
      return selected("pointwise");
    }
    if (name.rfind("q8ta_conv2d_dw", 0) == 0) {
      return selected("depthwise");
    }
    if (name.rfind("q8ta_conv2d_transposed", 0) == 0) {
      return selected("transposed");
    }
    if (name == "q8ta_conv2d_float" || name == "q8ta_conv2d_fallback_float") {
      return selected("direct");
    }
  }
  return Json();
}

REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d.default, direct, build_q8ta_conv2d_direct);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d.default, general, build_q8ta_conv2d_direct);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d.default, im2col, build_im2col_signed);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d.default, im2col_auto, build_q8ta_conv2d_im2col);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d.default, im2col_unsigned, build_im2col_unsigned);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d.default, im2col_fallback, build_im2col_fallback);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d.default, legacy_4w4c, build_legacy_4w4c);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d_dw.default, legacy_4w4c, build_legacy_4w4c);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d_pw.default, legacy_4w4c, build_legacy_4w4c);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d_pw.default, pw_signed, build_pw_signed);
  REGISTER_OPERATOR_ROUTE(
      et_vk.q8ta_conv2d_pw.default, pw_unsigned, build_pw_unsigned);
  REGISTER_OPERATOR_ROUTE(et_vk.q8ta_conv2d_pw.default, pw_auto, build_pw_auto);
  REGISTER_ROUTE_INSPECTOR(
      et_vk.q8ta_conv2d.default, inspect_q8ta_conv2d_route);
  REGISTER_ROUTE_INSPECTOR(
      et_vk.q8ta_conv2d_general.default, inspect_q8ta_conv2d_route);
  REGISTER_ROUTE_INSPECTOR(
      et_vk.q8ta_conv2d_im2col.default, inspect_q8ta_conv2d_route);
  REGISTER_ROUTE_INSPECTOR(
      et_vk.q8ta_conv2d_dw.default, inspect_q8ta_conv2d_route);
  REGISTER_ROUTE_INSPECTOR(
      et_vk.q8ta_conv2d_pw.default, inspect_q8ta_conv2d_route);
  REGISTER_ROUTE_INSPECTOR(
      et_vk.q8ta_conv2d_transposed.default, inspect_q8ta_conv2d_route);
}
} // namespace vkcompute::shader_profiling
