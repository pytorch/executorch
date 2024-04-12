/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <iostream>

namespace vkcompute {

void resize_conv2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr self = graph->get_tensor(args[1].refs[0]);

  size_t ndim = self->sizes().size();
  std::vector<int64_t> new_out_sizes(ndim);
  const bool transposed = graph->get_bool(extra_args[4]);

  // Batch, Channel
  if (ndim == 4) {
    new_out_sizes.at(ndim - 4) = self->sizes().at(ndim - 4);
  }

  TensorRefPtr weight_ref = graph->get_tref(extra_args[0]);
  const auto& weight_sizes = weight_ref->sizes;
  new_out_sizes.at(ndim - 3) =
      transposed ? weight_sizes.at(ndim - 3) : weight_sizes.at(ndim - 4);

  // Height, Width
  const auto& new_out_sizes_hw = calc_out_sizes_hw(
      *graph,
      self->sizes(),
      extra_args[0],
      /*kernel_size_only = */ false,
      {extra_args[1], extra_args[2], extra_args[3], extra_args[5]},
      transposed);
  new_out_sizes.at(ndim - 2) = new_out_sizes_hw.at(0);
  new_out_sizes.at(ndim - 1) = new_out_sizes_hw.at(1);

  out->virtual_resize(new_out_sizes);
}

ValueRef prepack_biases(
    ComputeGraph& graph,
    const ValueRef vref,
    const ValueRef weight,
    const bool transposed) {
  auto sizes = graph.get_sizes_of(weight);
  const int64_t out_channels = transposed ? sizes.at(1) : sizes.at(0);

  ValueRef v = graph.add_tensor(
      {out_channels},
      graph.get_dtype_of(weight),
      api::kTexture2D,
      api::kWidthPacked);
  vTensorPtr t = graph.get_tensor(v);

  api::ShaderInfo shader = get_nchw_to_image_shader(*t);

  api::utils::uvec3 global_size = t->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      global_size,
      local_size,
      vref,
      v,
      {t->gpu_sizes_ubo(), t->cpu_sizes_ubo()}));

  return v;
}

enum class Conv2dMethod : uint8_t {
  Depthwise,
  Pointwise,
  SlidingWindow,
  Transposed,
};

api::ShaderInfo get_conv2d_shader(
    ComputeGraph& graph,
    const vTensor& t_out,
    const bool prepack_weights,
    const Conv2dMethod method,
    const ValueRef weight) {
  std::string kernel_name;
  kernel_name.reserve(kShaderNameReserve);
  switch (method) {
    case Conv2dMethod::Depthwise:
      kernel_name = "conv2d_dw";
      if (!prepack_weights) {
        const auto& weight_sizes = graph.get_tref(weight)->sizes;
        if (weight_sizes.at(2) == 3 && weight_sizes.at(3) == 3) {
          kernel_name += "_output_tile_3x3";
        }
        if (weight_sizes.at(2) == 5 && weight_sizes.at(3) == 5) {
          kernel_name += "_output_tile_5x5";
        }
      }
      break;
    case Conv2dMethod::Pointwise:
      if (prepack_weights) {
        kernel_name = "conv2d";
      } else {
        kernel_name = "conv2d_pw";
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
  }
  add_dtype_suffix(kernel_name, t_out);

  return VK_KERNEL_FROM_STR(kernel_name);
}

std::vector<int64_t> get_final_sizes(
    const std::vector<int64_t>& original_sizes,
    const Conv2dMethod method) {
  int64_t batch_padded =
      api::utils::align_up(api::utils::val_at(-4, original_sizes), INT64_C(4));
  int64_t channels_padded =
      api::utils::align_up(api::utils::val_at(-3, original_sizes), INT64_C(4));
  int64_t channels = api::utils::val_at(-3, original_sizes);
  int64_t height = api::utils::val_at(-2, original_sizes);
  int64_t width = api::utils::val_at(-1, original_sizes);

  switch (method) {
    case Conv2dMethod::Depthwise:
      return std::vector<int64_t>{
          4, batch_padded * channels / 4, height * width};
    case Conv2dMethod::Pointwise:
    case Conv2dMethod::SlidingWindow:
      return std::vector<int64_t>{
          4, batch_padded * height / 4, channels_padded * width};
    case Conv2dMethod::Transposed:
      return std::vector<int64_t>{
          4, channels_padded * height / 4, batch_padded * width};
  }
}

std::vector<int64_t> get_padded_sizes(
    const std::vector<int64_t>& original_sizes,
    const Conv2dMethod method) {
  int64_t batch_padded =
      api::utils::align_up(api::utils::val_at(-4, original_sizes), INT64_C(4));
  int64_t channels_padded =
      api::utils::align_up(api::utils::val_at(-3, original_sizes), INT64_C(4));

  switch (method) {
    case Conv2dMethod::Depthwise:
      return std::vector<int64_t>{-1, batch_padded};
    case Conv2dMethod::Pointwise:
    case Conv2dMethod::SlidingWindow:
    case Conv2dMethod::Transposed:
      return std::vector<int64_t>{batch_padded, channels_padded};
  }
}

ValueRef prepack_weights(
    ComputeGraph& graph,
    const ValueRef vref,
    const Conv2dMethod method) {
  const auto original_sizes = graph.get_sizes_of(vref);
  const auto final_sizes = get_final_sizes(original_sizes, method);

  ValueRef v = graph.add_tensor(
      final_sizes,
      graph.get_dtype_of(vref),
      api::kTexture2D,
      api::kChannelsPacked);
  vTensorPtr t = graph.get_tensor(v);

  api::utils::uvec3 global_size = t->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  api::ShaderInfo shader =
      get_conv2d_shader(graph, *t, /*prepack_weights = */ true, method, vref);

  const auto& padded_sizes = get_padded_sizes(original_sizes, method);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      global_size,
      local_size,
      vref,
      v,
      {t->gpu_sizes_ubo(),
       graph.create_params_buffer(
           api::utils::make_ivec4(original_sizes, /*reverse = */ true)),
       graph.create_params_buffer(
           api::utils::make_ivec2(padded_sizes, /*reverse = */ true))}));

  return v;
}

void check_conv2d_args(const vTensor& in, const vTensor& out) {
  if (in.sizes().at(0) > 1) {
    VK_THROW(
        "aten.convolution.default: input batch size > 1 is not supported yet!");
  }
  VK_CHECK_COND(check_memory_layout_is(in, api::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, api::kChannelsPacked));
}

struct Conv2dParams final {
  api::utils::ivec2 overlay_region;
  int in_group_size;
};

Conv2dParams create_conv2d_params(
    ComputeGraph& graph,
    const ValueRef weight,
    const KernelParams& p,
    const bool transposed) {
  const auto& overlay_region = api::utils::make_ivec2({
      p.kernel_size.data[0] +
          (p.kernel_size.data[0] - 1) * (p.dilation.data[0] - 1),
      p.kernel_size.data[1] +
          (p.kernel_size.data[1] - 1) * (p.dilation.data[1] - 1),
  });
  const auto weight_sizes = graph.get_sizes_of(weight);
  const int32_t in_group_size =
      api::utils::safe_downcast<int32_t>(api::utils::align_up(
          transposed ? weight_sizes.at(0) : weight_sizes.at(1), INT64_C(4)));
  return {overlay_region, in_group_size};
}

void check_conv2d_params(const KernelParams& p, const bool transposed) {
  if (transposed) {
    if (p.dilation.data[0] > 1 || p.dilation.data[1] > 1) {
      VK_THROW(
          "aten.convolution.default: transposed = true, dilation > 1 is not supported yet!");
    }
  }
  if ((p.padding.data[0] > 0 && p.kernel_size.data[0] > 1 &&
       p.dilation.data[0] > 1) ||
      (p.padding.data[1] > 0 && p.kernel_size.data[1] > 1 &&
       p.dilation.data[1] > 1)) {
    VK_THROW(
        "aten.convolution.default: padding > 0 while dilation, kernel_size > 1 is not supported yet!");
  }
}

Conv2dMethod get_conv2d_method(
    ComputeGraph& graph,
    const ValueRef weight,
    const int64_t groups,
    const bool transposed) {
  const auto weight_sizes = graph.get_sizes_of(weight);
  if (!transposed && weight_sizes.at(0) == groups && weight_sizes.at(1) == 1) {
    return Conv2dMethod::Depthwise;
  }
  if (groups > 1) {
    VK_THROW("aten.convolution.default: groups > 1 is not supported yet!");
  }
  if (transposed) {
    return Conv2dMethod::Transposed;
  }
  if (weight_sizes.at(2) == 1 && weight_sizes.at(3) == 1) {
    return Conv2dMethod::Pointwise;
  }
  return Conv2dMethod::SlidingWindow;
}

void add_conv2d_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef transposed,
    const ValueRef output_padding,
    const ValueRef groups,
    const ValueRef out) {
  const bool transposed_val = graph.get_bool(transposed);
  const int64_t groups_val = graph.get_int(groups);

  const Conv2dMethod method =
      get_conv2d_method(graph, weight, groups_val, transposed_val);

  ValueRef arg_in = prepack_if_tensor_ref(graph, in);
  ValueRef arg_weight = prepack_weights(graph, weight, method);
  ValueRef arg_bias = prepack_biases(graph, bias, weight, transposed_val);

  vTensorPtr t_in = graph.get_tensor(arg_in);
  vTensorPtr t_out = graph.get_tensor(out);
  check_conv2d_args(*t_in, *t_out);

  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  KernelParams kernel_params = create_kernel_params(
      graph,
      weight,
      /*kernel_size_only = */ false,
      stride,
      padding,
      dilation);
  Conv2dParams extra_params =
      create_conv2d_params(graph, weight, kernel_params, transposed_val);

  check_conv2d_params(kernel_params, transposed_val);

  api::ShaderInfo shader = get_conv2d_shader(
      graph, *t_out, /*prepack_weights = */ false, method, weight);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE},
       {{arg_in, arg_weight, arg_bias}, api::MemoryAccessType::READ}},
      // Shader params buffers
      {
          t_out->extents_ubo(),
          t_in->extents_ubo(),
          graph.create_params_buffer(kernel_params),
          graph.create_params_buffer(extra_params),
      },
      // Resizing
      resize_conv2d_node,
      {weight, stride, padding, dilation, transposed, output_padding}));
}

void conv2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_conv2d_node(
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
      args[9]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.convolution.default, conv2d);
}

} // namespace vkcompute
