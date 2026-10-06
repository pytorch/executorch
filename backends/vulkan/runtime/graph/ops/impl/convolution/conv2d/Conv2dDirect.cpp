/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dDirect.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

vkapi::ShaderInfo get_conv2d_shader(
    ComputeGraph& graph,
    const ValueRef out,
    const bool prepack_weights,
    const Conv2dMethod method,
    const ValueRef weight,
    const bool clamp_out = false,
    const bool stride_equals_dilation = false,
    const bool stride_1_padding_0 = false) {
  (void)stride_equals_dilation;
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);
  switch (method) {
    case Conv2dMethod::Depthwise:
      kernel_name = "conv2d_dw";
      break;
    case Conv2dMethod::Pointwise:
      if (prepack_weights) {
        kernel_name = "conv2d";
      } else {
        kernel_name = stride_1_padding_0 ? "conv2d_pw_s1p0" : "conv2d_pw";
      }
      break;
    case Conv2dMethod::SlidingWindow:
      kernel_name = "conv2d";
      break;
    case Conv2dMethod::Transposed:
      kernel_name = "conv_transpose2d";
      break;
  }
  if (prepack_weights) {
    kernel_name += "_prepack_weights";
  } else if (clamp_out) {
    kernel_name += "_clamp";
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  if (prepack_weights) {
    add_dtype_suffix(kernel_name, graph.get_staging_dtype_for(weight));
  }

  return VK_KERNEL_FROM_STR(kernel_name);
}

std::vector<int64_t> get_final_sizes(
    const std::vector<int64_t>& original_sizes,
    const Conv2dMethod method) {
  int64_t batch_padded = utils::align_up_4(utils::val_at(-4, original_sizes));
  int64_t channels_padded =
      utils::align_up_4(utils::val_at(-3, original_sizes));
  int64_t height = utils::val_at(-2, original_sizes);
  int64_t width = utils::val_at(-1, original_sizes);

  switch (method) {
    case Conv2dMethod::Depthwise:
      return std::vector<int64_t>{4, batch_padded / 4, height * width};
    case Conv2dMethod::Pointwise:
    case Conv2dMethod::SlidingWindow:
      return std::vector<int64_t>{
          4, batch_padded * height / 4, channels_padded * width};
    case Conv2dMethod::Transposed:
      return std::vector<int64_t>{
          4, channels_padded * height / 4, batch_padded * width};
  }
}

ValueRef prepack_weights(
    ComputeGraph& graph,
    const ValueRef vref,
    const Conv2dMethod method) {
  const auto original_sizes = graph.sizes_of(vref);
  const auto final_sizes = get_final_sizes(original_sizes, method);

  ValueRef v = graph.add_tensor(
      final_sizes,
      graph.dtype_of(vref),
      utils::kTexture2D,
      utils::kChannelsPacked);

  vkapi::ShaderInfo shader =
      get_conv2d_shader(graph, v, /*prepack_weights = */ true, method, vref);

  const auto original_sizes_pc =
      utils::make_ivec4(original_sizes, /*reverse = */ true);
  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_gwg(v),
      graph.create_lwg(v),
      vref,
      v,
      {},
      // Specialization constants
      {graph.packed_dim_of(v)},
      {graph.sizes_pc_of(v),
       PushConstantDataInfo(&original_sizes_pc, sizeof(original_sizes_pc))}));

  return v;
}

namespace {

struct Conv2dParams final {
  utils::ivec2 overlay_region;
  int in_group_size;
};

} // namespace

struct OutputParams final {
  float out_min;
  float out_max;
};

Conv2dParams create_conv2d_params(
    ComputeGraph& graph,
    const ValueRef weight,
    const Kernel2dParams& p,
    const bool transposed) {
  const auto& overlay_region = utils::make_ivec2({
      p.kernel_size[0] + (p.kernel_size[0] - 1) * (p.dilation[0] - 1),
      p.kernel_size[1] + (p.kernel_size[1] - 1) * (p.dilation[1] - 1),
  });
  const auto weight_sizes = graph.sizes_of(weight);
  const int32_t in_group_size = utils::safe_downcast<int32_t>(
      utils::align_up_4(transposed ? weight_sizes.at(0) : weight_sizes.at(1)));
  return {overlay_region, in_group_size};
}

void check_conv2d_params(const Kernel2dParams& p, const bool transposed) {
  if (transposed) {
    if (p.dilation[0] > 1 || p.dilation[1] > 1) {
      VK_THROW(
          "aten.convolution.default: transposed = true, dilation > 1 is not supported yet!");
    }
  }
}

GlobalWorkGrid create_conv2d_gwg(
    ComputeGraph& graph,
    const Conv2dMethod method,
    const ValueRef out,
    const ValueRef weight_data,
    const bool stride_equals_dilation) {
  (void)weight_data;
  (void)stride_equals_dilation;
  if (method == Conv2dMethod::Pointwise) {
    const utils::uvec3 image_extents = graph.logical_limits_of(out);
    return GlobalWorkGrid(
        {utils::div_up(image_extents[0u], 1u),
         utils::div_up(image_extents[1u], 4u),
         image_extents[2u]},
        kTiledWorkGrid);
  } else {
    return graph.create_gwg(out);
  }
}

// Determines which convolution method a dispatch uses.
//
// Depthwise and transposed convolutions have shader names of their own, but
// the name alone cannot separate pointwise from sliding window: the sliding
// window shader is itself named "conv2d", and a pointwise convolution also
// takes that name when its weights are prepacked. Those two are therefore
// separated by the weight's spatial extent. Shared by the global and local
// workgroup size functions below so that the two cannot disagree about the
// same dispatch.
Conv2dMethod infer_conv2d_method_from_shader(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const ValueRef weight_data) {
  const std::string& kernel_name = shader.kernel_name;
  // Checked before the plain "conv2d" test below, which "conv2d_dw" and
  // "conv2d_pw" would otherwise match too.
  if (kernel_name.find("conv2d_dw") != std::string::npos) {
    return Conv2dMethod::Depthwise;
  }
  if (kernel_name.find("conv2d_pw") != std::string::npos) {
    return Conv2dMethod::Pointwise;
  }
  if (kernel_name.find("conv_transpose2d") != std::string::npos) {
    return Conv2dMethod::Transposed;
  }
  const auto& weight_sizes = graph->get_tref(weight_data)->sizes;
  if (weight_sizes.at(2) == 1 && weight_sizes.at(3) == 1) {
    return Conv2dMethod::Pointwise;
  }
  return Conv2dMethod::SlidingWindow;
}

// Custom global workgroup size function for conv2d
GlobalWorkGrid conv2d_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef weight_data = resize_args.at(0);

  const Conv2dMethod method =
      infer_conv2d_method_from_shader(graph, shader, weight_data);

  // Determine stride_equals_dilation from shader name
  bool stride_equals_dilation =
      shader.kernel_name.find("_sned") == std::string::npos;

  const GlobalWorkGrid wg_size = create_conv2d_gwg(
      *graph, method, out, weight_data, stride_equals_dilation);

  if (method == Conv2dMethod::Pointwise) {
    utils::uvec3 pointwise_wg_size = {wg_size[0] * wg_size[1], wg_size[2], 1u};

    if (shader.kernel_name.find("s1p0") != std::string::npos) {
      pointwise_wg_size[0] *= 4;
    }
    return GlobalWorkGrid(pointwise_wg_size, kTiledWorkGrid);
  }

  return wg_size;
}

// Custom local workgroup size function for conv2d
LocalWorkGroup conv2d_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;

  const ValueRef weight_data = resize_args.at(0);
  const Conv2dMethod method =
      infer_conv2d_method_from_shader(graph, shader, weight_data);

  if (method == Conv2dMethod::Pointwise) {
    uint32_t lwg_y = 1;
    if (gwg[1] % 8 == 0) {
      lwg_y = 8;
    } else if (gwg[1] % 4 == 0) {
      lwg_y = 4;
    } else if (gwg[1] % 2 == 0) {
      lwg_y = 2;
    }
    return LocalWorkGroup(64u / lwg_y, lwg_y, 1u);
  } else {
    return graph->create_lwg(gwg);
  }
}

void add_conv2d_direct_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef arg_weight,
    const ValueRef arg_bias,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef transposed,
    const ValueRef output_padding,
    const ValueRef out,
    const vkapi::ShaderInfo& shader,
    const Kernel2dParams& kernel_params,
    const Conv2dParams& extra_params,
    const OutputParams& out_params,
    const int64_t groups_val) {
  vkapi::ParamsBindList param_buffers = {
      graph.logical_limits_ubo(out),
      graph.sizes_ubo(in),
      graph.create_params_buffer(kernel_params),
      graph.create_params_buffer(extra_params),
      graph.create_params_buffer(out_params),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      shader,
      conv2d_gwg,
      conv2d_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {utils::safe_downcast<int32_t>(groups_val)},
      // Resize Args
      {weight_data, stride, padding, dilation, transposed, output_padding},
      // Resizing Logic
      resize_conv2d_node));
}

void conv2d_direct_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef transposed,
    const ValueRef output_padding,
    const ValueRef groups,
    const ValueRef out,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val) {
  const bool transposed_val = graph.get_bool(transposed);
  const int64_t groups_val = graph.get_int(groups);
  const Conv2dMethod method =
      transposed_val ? Conv2dMethod::Transposed : Conv2dMethod::SlidingWindow;

  const Kernel2dParams kernel_params = create_kernel2d_params(
      graph,
      weight_data,
      /*kernel_size_only = */ false,
      stride,
      padding,
      dilation);

  ValueRef arg_weight = prepack_weights(graph, weight_data, method);
  ValueRef arg_bias = prepack_biases(
      graph,
      bias,
      weight_data,
      transposed_val,
      /* storage_type = */ utils::kTexture2D,
      /* memory_layout = */ utils::kWidthPacked);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  if (in_sizes.at(0) > 1) {
    VK_THROW("conv2d: input batch size > 1 is not supported yet!");
  }

  check_conv_args(graph, in, out);

  Conv2dParams extra_params =
      create_conv2d_params(graph, weight_data, kernel_params, transposed_val);

  const bool stride_equals_dilation =
      (kernel_params.stride[0] == kernel_params.dilation[0] &&
       kernel_params.stride[1] == kernel_params.dilation[1]);

  const bool stride_1_padding_0 =
      (kernel_params.stride[0] == 1 && kernel_params.stride[1] == 1 &&
       kernel_params.padding[0] == 0 && kernel_params.padding[1] == 0);

  OutputParams out_params = {out_min_val, out_max_val};

  check_conv2d_params(kernel_params, transposed_val);

  vkapi::ShaderInfo shader = get_conv2d_shader(
      graph,
      out,
      /*prepack_weights = */ false,
      method,
      weight_data,
      clamp_out,
      stride_equals_dilation,
      stride_1_padding_0);

  add_conv2d_direct_node(
      graph,
      in,
      arg_weight,
      arg_bias,
      weight_data,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      out,
      shader,
      kernel_params,
      extra_params,
      out_params,
      groups_val);
}

} // namespace vkcompute
