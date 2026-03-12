/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv1d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_conv1d_pw_buf_node(
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

  std::string kernel_name = clamp_out ? "conv1d_pw_clamp" : "conv1d_pw";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      conv1d_buf_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers (UBOs) - must match conv1d_pw.glsl binding order
      {
          graph.meta_ubo(out),
          graph.meta_ubo(in),
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

static ValueRef prepack_conv1d_pw_weight_buffer(
    ComputeGraph& graph,
    const ValueRef vref) {
  const auto sizes = graph.sizes_of(vref);
  // Weight [C_out, C_in/g, 1] -> flatten to [C_out * C_in/g] buffer
  const int64_t numel = sizes.at(0) * sizes.at(1);
  ValueRef v = graph.add_tensor(
      {numel}, graph.dtype_of(vref), utils::kBuffer, utils::kWidthPacked);

  vkapi::ShaderInfo shader =
      get_nchw_to_tensor_shader(graph, v, graph.get_staging_dtype_for(vref));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      graph.create_global_wg_size(v),
      graph.create_local_wg_size(v),
      vref,
      v,
      {graph.buffer_meta_ubo(v)},
      {graph.hashed_layout_of(v), 0},
      {graph.sizes_pc_of(v), graph.strides_pc_of(v), graph.numel_pc_of(v)}));

  return v;
}

void add_conv1d_pw_texture_node(
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
  const int64_t out_channels = weight_sizes.at(0);
  const int64_t in_group_size = weight_sizes.at(1);
  const int64_t out_group_size = out_channels / groups_val;

  ValueRef arg_weight = prepack_conv1d_pw_weight_buffer(graph, weight_data);
  ValueRef arg_bias =
      prepack_conv1d_bias(graph, bias, weight_data, out_channels);

  struct ConvParams {
    int32_t in_group_size;
    int32_t out_group_size;
  } conv_params{
      static_cast<int32_t>(in_group_size),
      static_cast<int32_t>(out_group_size),
  };

  struct OutputParams {
    float out_min;
    float out_max;
  } out_params{out_min_val, out_max_val};

  std::string kernel_name =
      clamp_out ? "conv1d_pw_texture_clamp" : "conv1d_pw_texture";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      // Global workgroup size: (L_out_texels, C_out, N)
      [](ComputeGraph* g,
         const vkapi::ShaderInfo&,
         const std::vector<ArgGroup>& a,
         const std::vector<ValueRef>&) -> utils::uvec3 {
        const ValueRef o = a.at(0).refs.at(0);
        const auto limits = g->logical_limits_of(o);
        return {
            static_cast<uint32_t>(limits[0]),
            static_cast<uint32_t>(limits[1]),
            static_cast<uint32_t>(limits[2])};
      },
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // UBOs - must match conv1d_pw_texture.glsl binding order
      {
          graph.logical_limits_ubo(out),
          graph.create_params_buffer(conv_params),
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

} // namespace vkcompute
