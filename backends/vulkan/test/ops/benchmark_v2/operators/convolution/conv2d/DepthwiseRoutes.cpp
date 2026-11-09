// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

struct Conv2dDWParams final {
  utils::ivec2 overlay_region;
  int in_group_size;
};

struct OutputParams final {
  float out_min;
  float out_max;
};

static std::string pick_conv2d_dw_shader_with_selector(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const ValueRef out,
    const bool stride_equals_dilation,
    const bool clamp_out,
    const std::string& impl_selector) {
  std::string kernel_name = "conv2d_dw";
  kernel_name.reserve(40);

  const auto& weight_sizes = graph.get_tref(weight_data)->sizes;
  const bool is_3x3 = weight_sizes.at(2) == 3 && weight_sizes.at(3) == 3;
  const bool is_5x5 = weight_sizes.at(2) == 5 && weight_sizes.at(3) == 5;

  if (!stride_equals_dilation && (is_3x3 || is_5x5)) {
    kernel_name += "_sned";
  }

  if (is_3x3) {
    kernel_name += "_output_tile_3x3";
    if (impl_selector == "b1x1") {
      if (stride_equals_dilation) {
        kernel_name += "_b1x1";
      }
    } else if (impl_selector == "b4x2") {
    } else {
      if (stride_equals_dilation && graph.device_is_mali()) {
        kernel_name += "_b1x1";
      }
    }
  } else if (is_5x5) {
    kernel_name += "_output_tile_5x5";
  }

  if (clamp_out) {
    kernel_name += "_clamp";
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  return kernel_name;
}

static GlobalWorkGrid conv2d_dw_gwg_fn(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  const bool uses_output_tile =
      shader.kernel_name.find("_output_tile") != std::string::npos;

  if (uses_output_tile) {
    const bool is_sned = shader.kernel_name.find("_sned") != std::string::npos;
    const utils::uvec3 image_extents = graph->create_gwg(out).extents();

    if (is_sned) {
      return GlobalWorkGrid(
          {image_extents[0] * image_extents[1], image_extents[2], 1u},
          kTiledWorkGrid);
    }

    uint32_t batch_x = 4u;
    uint32_t batch_y = 2u;
    if (shader.kernel_name.find("_b1x1") != std::string::npos) {
      batch_x = 1u;
      batch_y = 1u;
    }

    uint32_t scaled_x = utils::div_up(image_extents[0], batch_x);
    uint32_t scaled_y = utils::div_up(image_extents[1], batch_y);
    return GlobalWorkGrid(
        {scaled_x * scaled_y, image_extents[2], 1u}, kTiledWorkGrid);
  }

  const utils::uvec3 base_extents = graph->create_gwg(out).extents();
  return graph->create_linear_gwg(
      base_extents[0] * base_extents[1] * base_extents[2]);
}

static LocalWorkGroup conv2d_dw_lwg_fn(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)gwg;
  (void)args;
  (void)resize_args;
  return LocalWorkGroup(64u, 1u, 1u);
}

static ValueRef prepack_dw_weights(ComputeGraph& graph, const ValueRef vref) {
  const auto original_sizes = graph.sizes_of(vref);

  int64_t out_channels_padded =
      utils::align_up_4(utils::val_at(-4, original_sizes));
  int64_t height = utils::val_at(-2, original_sizes);
  int64_t width = utils::val_at(-1, original_sizes);

  const std::vector<int64_t> final_sizes = {
      4, out_channels_padded / 4, height * width};

  ValueRef v = graph.add_tensor(
      final_sizes,
      graph.dtype_of(vref),
      utils::kTexture2D,
      utils::kChannelsPacked);

  std::string kernel_name = "conv2d_dw_prepack_weights";
  add_dtype_suffix(kernel_name, graph.dtype_of(v));
  add_dtype_suffix(kernel_name, graph.get_staging_dtype_for(vref));

  const auto original_sizes_pc = utils::make_ivec4(original_sizes, true);
  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_gwg(v),
      graph.create_lwg(v),
      vref,
      v,
      {},
      {graph.packed_dim_of(v)},
      {graph.sizes_pc_of(v),
       PushConstantDataInfo(&original_sizes_pc, sizeof(original_sizes_pc))}));

  return v;
}

static void conv2d_dw_with_selector(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const std::string& impl_selector) {
  ValueRef arg_weight = prepack_dw_weights(graph, weight_data);
  ValueRef arg_bias = prepack_biases(
      graph, bias, weight_data, false, utils::kTexture2D, utils::kWidthPacked);

  check_conv_args(graph, in, out);

  Kernel2dParams kernel_params = create_kernel2d_params(
      graph, weight_data, false, stride, padding, dilation);

  const bool stride_equals_dilation =
      (kernel_params.stride[0] == kernel_params.dilation[0] &&
       kernel_params.stride[1] == kernel_params.dilation[1]);

  const auto& overlay_region = utils::make_ivec2({
      kernel_params.kernel_size[0] +
          (kernel_params.kernel_size[0] - 1) * (kernel_params.dilation[0] - 1),
      kernel_params.kernel_size[1] +
          (kernel_params.kernel_size[1] - 1) * (kernel_params.dilation[1] - 1),
  });
  const auto weight_sizes = graph.sizes_of(weight_data);
  const int32_t in_group_size =
      utils::safe_downcast<int32_t>(utils::align_up_4(weight_sizes.at(1)));
  Conv2dDWParams extra_params = {overlay_region, in_group_size};

  OutputParams out_params = {
      std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()};

  std::string kernel_name = pick_conv2d_dw_shader_with_selector(
      graph, weight_data, out, stride_equals_dilation, false, impl_selector);

  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers;
  std::vector<PushConstantDataInfo> push_constants;

  const bool uses_output_tile =
      kernel_name.find("_output_tile") != std::string::npos;

  if (uses_output_tile) {
    const utils::ivec4 kernel_param_size_stride = {
        kernel_params.kernel_size[0],
        kernel_params.kernel_size[1],
        kernel_params.stride[0],
        kernel_params.stride[1]};

    const utils::ivec4 kernel_param_pad_dial = {
        kernel_params.padding[0],
        kernel_params.padding[1],
        kernel_params.dilation[0],
        kernel_params.dilation[1]};

    push_constants = {
        graph.logical_limits_pc_of(out),
        graph.sizes_pc_of(in),
        PushConstantDataInfo(
            &kernel_param_size_stride, sizeof(kernel_param_size_stride)),
        PushConstantDataInfo(
            &kernel_param_pad_dial, sizeof(kernel_param_pad_dial)),
        PushConstantDataInfo(
            &extra_params, sizeof(extra_params), sizeof(utils::ivec4)),
        PushConstantDataInfo(&out_params, sizeof(out_params)),
    };
  } else {
    param_buffers = {
        graph.logical_limits_ubo(out),
        graph.sizes_ubo(in),
        graph.create_params_buffer(kernel_params),
        graph.create_params_buffer(extra_params),
        graph.create_params_buffer(out_params),
    };
  }

  ValueRef transposed_ref = graph.add_scalar(false);
  ValueRef output_padding = graph.add_none();

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      shader,
      conv2d_dw_gwg_fn,
      conv2d_dw_lwg_fn,
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      param_buffers,
      push_constants,
      {},
      {weight_data, stride, padding, dilation, transposed_ref, output_padding},
      resize_conv2d_node));
}

namespace shader_profiling {
namespace {
template <bool B1>
Json build_dw_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options) {
  if (!options.empty())
    throw std::invalid_argument("Depthwise tile routes accept no options");
  if (args.size() != 10 || graph.extract_scalar<bool>(args[6]) ||
      graph.extract_scalar<int64_t>(args[8]) != graph.sizes_of(args[0]).at(1))
    throw UnsupportedRoute(
        "Depthwise tile routes require forward depthwise convolution");
  if (!graph.is_texture_storage(args[0]) ||
      !graph.is_texture_storage(args.back()))
    throw UnsupportedRoute("Depthwise tile routes require texture IO");
  conv2d_dw_with_selector(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args[9],
      B1 ? "b1x1" : "b4x2");
  const auto& sizes = graph.sizes_of(args[1]);
  const auto stride = *graph.get_int_list(args[3]);
  const auto dilation = *graph.get_int_list(args[5]);
  const bool selected =
      B1 && sizes[2] == 3 && sizes[3] == 3 && stride == dilation;
  return {{"name", selected ? "b1x1" : "b4x2"}, {"options", options}};
}
REGISTER_OPERATOR_ROUTES {
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, b1x1, (build_dw_route<true>));
  REGISTER_OPERATOR_ROUTE(
      aten.convolution.default, b4x2, (build_dw_route<false>));
}
} // namespace
} // namespace shader_profiling
} // namespace vkcompute
