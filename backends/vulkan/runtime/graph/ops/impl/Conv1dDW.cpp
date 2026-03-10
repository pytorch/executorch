/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv1d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_conv1d_dw_buf_node(
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

  std::string kernel_name = clamp_out ? "conv1d_dw_clamp" : "conv1d_dw";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      conv1d_buf_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers (UBOs) - must match conv1d_dw.glsl binding order
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

void add_conv1d_dw_texture_node(
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
  struct ConvDWParams {
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
    int32_t dilation;
  } conv_dw_params{
      kernel_params.kernel_size,
      kernel_params.stride,
      kernel_params.padding,
      kernel_params.dilation,
  };

  struct InLengthParams {
    int32_t in_length;
  } in_length_params{
      graph.size_at<int32_t>(-1, in),
  };

  struct OutputParams {
    float out_min;
    float out_max;
  } out_params{out_min_val, out_max_val};

  std::string kernel_name =
      clamp_out ? "conv1d_dw_texture_clamp" : "conv1d_dw_texture";
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
      // UBOs - must match conv1d_dw_texture.glsl binding order
      {
          graph.logical_limits_ubo(out),
          graph.create_params_buffer(conv_dw_params),
          graph.create_params_buffer(in_length_params),
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
