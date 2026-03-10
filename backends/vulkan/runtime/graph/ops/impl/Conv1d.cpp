/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv1d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

enum class Conv1dMethod : uint8_t {
  Depthwise,
  Pointwise,
  General,
};

void resize_conv1d_buf_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));

  const int64_t stride_size = graph->get_int_list(extra_args.at(1))->at(0);
  const int64_t padding_size = graph->get_int_list(extra_args.at(2))->at(0);
  const int64_t dilation_size = graph->get_int_list(extra_args.at(3))->at(0);

  const std::vector<int64_t>& weight_sizes = weight_ref->sizes;
  const std::vector<int64_t> in_sizes = graph->sizes_of(self);
  const size_t ndim = in_sizes.size();
  std::vector<int64_t> new_out_sizes(ndim);

  const int64_t kernel_size = weight_sizes.at(2);
  const int64_t in_length = in_sizes.at(2);

  new_out_sizes.at(0) = in_sizes.at(0);
  new_out_sizes.at(1) = weight_sizes.at(0);
  new_out_sizes.at(2) = calc_out_size(
      in_length, kernel_size, stride_size, padding_size, dilation_size, false);

  graph->virtual_resize(out, new_out_sizes);
}

ValueRef prepack_conv1d_bias(
    ComputeGraph& graph,
    const ValueRef vref,
    const ValueRef weight_data,
    const int64_t out_channels) {
  ValueRef v = graph.add_tensor(
      {out_channels},
      graph.dtype_of(weight_data),
      utils::kBuffer,
      utils::kWidthPacked);

  // Use staging dtype from weight (vref may be None for bias=None).
  vkapi::ShaderInfo shader = get_nchw_to_tensor_shader(
      graph, v, graph.get_staging_dtype_for(weight_data));

  // Must match add_prepack_standard_node's bindings for buffer-backed tensors.
  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_global_wg_size(v),
      graph.create_local_wg_size(v),
      vref,
      v,
      // Parameter Buffers
      {graph.buffer_meta_ubo(v)},
      // Specialization Constants: layout hash + transpose_hw=0
      {graph.hashed_layout_of(v), 0},
      // Push Constants: sizes, strides, numel
      {graph.sizes_pc_of(v), graph.strides_pc_of(v), graph.numel_pc_of(v)}));

  return v;
}

utils::uvec3 conv1d_buf_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return {
      graph->size_at<uint32_t>(-1, out), // L_out
      graph->size_at<uint32_t>(-2, out), // C_out
      graph->size_at<uint32_t>(-3, out), // N
  };
}

static void add_conv1d_general_buf_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef arg_weight,
    const ValueRef arg_bias,
    const ValueRef out,
    const Kernel1dParams& kernel_params,
    const float out_min_val,
    const float out_max_val,
    const bool clamp_out,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation) {
  struct OutputParams {
    float out_min;
    float out_max;
  } out_params{out_min_val, out_max_val};

  std::string kernel_name = clamp_out ? "conv1d_clamp" : "conv1d";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      conv1d_buf_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers (UBOs) - must match conv1d.glsl binding order
      {
          graph.sizes_ubo(out),
          graph.strides_ubo(out),
          graph.sizes_ubo(in),
          graph.strides_ubo(in),
          graph.strides_ubo(arg_weight),
          graph.create_params_buffer(kernel_params),
          graph.create_params_buffer(out_params),
      },
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {weight_data, stride, padding, dilation},
      // Resizing Logic
      resize_conv1d_buf_node));
}

static Conv1dMethod get_conv1d_method(
    const std::vector<int64_t>& weight_sizes,
    const int64_t groups) {
  if (weight_sizes.at(0) == groups && weight_sizes.at(1) == 1) {
    return Conv1dMethod::Depthwise;
  }
  if (weight_sizes.at(2) == 1) {
    return Conv1dMethod::Pointwise;
  }
  return Conv1dMethod::General;
}

void add_conv1d_dw_texture_entry(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const float out_min_val,
    const float out_max_val,
    const ValueRef out,
    const bool clamp_out) {
  const int64_t groups_val = graph.get_int(groups);
  const auto weight_sizes = graph.sizes_of(weight_data);

  ValueRef arg_weight =
      prepack_standard(graph, weight_data, utils::kBuffer, utils::kWidthPacked);

  const int64_t out_channels = weight_sizes.at(0);
  ValueRef arg_bias =
      prepack_conv1d_bias(graph, bias, weight_data, out_channels);

  const Kernel1dParams kernel_params = {
      static_cast<int>(weight_sizes.at(2)), // kernel_size
      static_cast<int>(graph.get_int_list(stride)->at(0)),
      static_cast<int>(graph.get_int_list(padding)->at(0)),
      static_cast<int>(graph.get_int_list(dilation)->at(0)),
      static_cast<int>(weight_sizes.at(1)), // in_group_size = C_in/groups (=1)
      static_cast<int>(out_channels / groups_val), // out_group_size (=1)
  };

  add_conv1d_dw_texture_node(
      graph,
      in,
      arg_weight,
      arg_bias,
      out,
      kernel_params,
      out_min_val,
      out_max_val,
      clamp_out,
      weight_data,
      stride,
      padding,
      dilation);
}

void add_conv1d_buf_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const float out_min_val,
    const float out_max_val,
    const ValueRef out,
    const bool clamp_out) {
  const int64_t groups_val = graph.get_int(groups);
  const auto weight_sizes = graph.sizes_of(weight_data);

  ValueRef arg_weight =
      prepack_standard(graph, weight_data, utils::kBuffer, utils::kWidthPacked);

  const int64_t out_channels = weight_sizes.at(0);
  ValueRef arg_bias =
      prepack_conv1d_bias(graph, bias, weight_data, out_channels);

  const Kernel1dParams kernel_params = {
      static_cast<int>(weight_sizes.at(2)), // kernel_size
      static_cast<int>(graph.get_int_list(stride)->at(0)),
      static_cast<int>(graph.get_int_list(padding)->at(0)),
      static_cast<int>(graph.get_int_list(dilation)->at(0)),
      static_cast<int>(weight_sizes.at(1)), // in_group_size = C_in/groups
      static_cast<int>(out_channels / groups_val), // out_group_size
  };

  const Conv1dMethod method = get_conv1d_method(weight_sizes, groups_val);

  switch (method) {
    case Conv1dMethod::Depthwise:
      add_conv1d_dw_buf_node(
          graph,
          in,
          arg_weight,
          arg_bias,
          out,
          kernel_params,
          out_min_val,
          out_max_val,
          clamp_out,
          weight_data,
          stride,
          padding,
          dilation);
      break;
    case Conv1dMethod::Pointwise:
      add_conv1d_pw_buf_node(
          graph,
          in,
          arg_weight,
          arg_bias,
          out,
          kernel_params,
          out_min_val,
          out_max_val,
          clamp_out,
          weight_data,
          stride,
          padding,
          dilation);
      break;
    case Conv1dMethod::General:
      add_conv1d_general_buf_node(
          graph,
          in,
          arg_weight,
          arg_bias,
          out,
          kernel_params,
          out_min_val,
          out_max_val,
          clamp_out,
          weight_data,
          stride,
          padding,
          dilation);
      break;
  }
}

} // namespace vkcompute
