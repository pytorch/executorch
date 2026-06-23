/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Local copies of Conv2dDW internals, extended with impl_selector support.
// These mirror the logic in Conv2dDW.cpp but allow forcing a specific tile size
// variant via the impl_selector string.
//

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

  if (!stride_equals_dilation) {
    kernel_name += "_sned";
  }

  if (is_3x3) {
    kernel_name += "_output_tile_3x3";
    if (impl_selector == "b1x1") {
      kernel_name += "_b1x1";
    } else if (impl_selector == "b4x2") {
      // b4x2 is the default (no suffix)
    } else {
      // Auto-selection: use b1x1 on Mali
      if (stride_equals_dilation && graph.device_is_mali()) {
        kernel_name += "_b1x1";
      }
    }
  } else if (is_5x5) {
    kernel_name += "_output_tile_5x5";
    // No b1x1 variant for 5x5; impl_selector is ignored for batch size
  }

  if (clamp_out) {
    kernel_name += "_clamp";
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  return kernel_name;
}

static utils::uvec3 conv2d_dw_global_wg_size_fn(
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
    const utils::uvec3 image_extents = graph->create_global_wg_size(out);

    if (is_sned) {
      return {image_extents[0] * image_extents[1], image_extents[2], 1};
    }

    uint32_t batch_x = 4u;
    uint32_t batch_y = 2u;
    if (shader.kernel_name.find("_b1x1") != std::string::npos) {
      batch_x = 1u;
      batch_y = 1u;
    }

    uint32_t scaled_x = utils::div_up(image_extents[0], batch_x);
    uint32_t scaled_y = utils::div_up(image_extents[1], batch_y);
    return {scaled_x * scaled_y, image_extents[2], 1};
  }

  const utils::uvec3 base_extents = graph->create_global_wg_size(out);
  return {base_extents[0] * base_extents[1] * base_extents[2], 1, 1};
}

static utils::uvec3 conv2d_dw_local_wg_size_fn(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {64, 1, 1};
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

  const auto original_sizes_pc =
      utils::make_ivec4(original_sizes, /*reverse=*/true);
  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(v),
      graph.create_local_wg_size(v),
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
      graph,
      bias,
      weight_data,
      /*transposed=*/false,
      /*storage_type=*/utils::kTexture2D,
      /*memory_layout=*/utils::kWidthPacked);

  check_conv_args(graph, in, out);

  Kernel2dParams kernel_params = create_kernel2d_params(
      graph,
      weight_data,
      /*kernel_size_only=*/false,
      stride,
      padding,
      dilation);

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
      graph,
      weight_data,
      out,
      stride_equals_dilation,
      /*clamp_out=*/false,
      impl_selector);

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
      conv2d_dw_global_wg_size_fn,
      conv2d_dw_local_wg_size_fn,
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      param_buffers,
      push_constants,
      {},
      {weight_data, stride, padding, dilation, transposed_ref, output_padding},
      resize_conv2d_node));
}

void test_conv2d_dw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[0] = input [N, C, H, W]
  // args[1] = weight [C, 1, K_h, K_w] (constant)
  // args[2] = bias (constant, or none)
  // args[3] = stride_h (int)
  // args[4] = stride_w (int)
  // args[5] = padding_h (int)
  // args[6] = padding_w (int)
  // args[7] = dilation_h (int)
  // args[8] = dilation_w (int)
  // args[9] = impl_selector (string)
  // args[10] = output
  const ValueRef input = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const int64_t stride_h = graph.extract_scalar<int64_t>(args.at(3));
  const int64_t stride_w = graph.extract_scalar<int64_t>(args.at(4));
  const int64_t padding_h = graph.extract_scalar<int64_t>(args.at(5));
  const int64_t padding_w = graph.extract_scalar<int64_t>(args.at(6));
  const int64_t dilation_h = graph.extract_scalar<int64_t>(args.at(7));
  const int64_t dilation_w = graph.extract_scalar<int64_t>(args.at(8));
  const std::string impl_selector = graph.extract_string(args.at(9));
  const ValueRef out = args.at(10);

  ValueRef stride =
      graph.add_scalar_list<int64_t>(std::vector<int64_t>{stride_h, stride_w});
  ValueRef padding = graph.add_scalar_list<int64_t>(
      std::vector<int64_t>{padding_h, padding_w});
  ValueRef dilation = graph.add_scalar_list<int64_t>(
      std::vector<int64_t>{dilation_h, dilation_w});

  if (impl_selector.empty()) {
    // Auto-selection: delegate to aten.convolution.default
    const int64_t channels = graph.sizes_of(input).at(1);
    ValueRef transposed = graph.add_scalar<bool>(false);
    ValueRef output_padding =
        graph.add_scalar_list<int64_t>(std::vector<int64_t>{0, 0});
    ValueRef groups = graph.add_scalar<int64_t>(channels);

    VK_GET_OP_FN("aten.convolution.default")
    (graph,
     {input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      out});
  } else {
    // Forced variant: build the dispatch directly with impl_selector
    conv2d_dw_with_selector(
        graph,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        out,
        impl_selector);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_conv2d_dw.default, test_conv2d_dw);
}

} // namespace vkcompute
