/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv1d/Conv1dDirect.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/Convolution.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_conv1d_node(
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

struct OutputParams final {
  float out_min;
  float out_max;
};

// Custom global workgroup size function for conv1d
GlobalWorkGrid conv1d_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  return GlobalWorkGrid(
      {// out length
       graph->size_at<uint32_t>(-1, out),
       // out channels
       static_cast<uint32_t>(graph->size_at<int64_t>(-2, out)),
       // out batches
       utils::div_up_4(graph->size_at<uint32_t>(-3, out))},
      kTiledWorkGrid);
}

void add_conv1d_direct_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef out_min,
    const ValueRef out_max,
    const ValueRef out,
    const bool clamp_out) {
  ValueRef arg_weight = prepack_standard(
      graph,
      weight,
      graph.storage_type_of(out),
      utils::kChannelsPacked,
      /* passthrough = */ false);
  ValueRef arg_bias = prepack_biases(
      graph,
      bias,
      weight,
      /*transposed = */ false,
      /*storage_type = */ utils::kTexture3D,
      /*memory_layout = */ utils::kWidthPacked);

  float out_min_val = 0.0f;
  float out_max_val = 0.0f;
  if (out_min != kDummyValueRef) {
    out_min_val = graph.extract_scalar<float>(out_min);
  }
  if (out_max != kDummyValueRef) {
    out_max_val = graph.extract_scalar<float>(out_max);
  }

  const int64_t groups_val = graph.get_int(groups);

  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> weight_sizes = graph.sizes_of(arg_weight);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);

  check_conv_args(graph, in, out);

  const int32_t in_channels = utils::safe_downcast<int32_t>(in_sizes.at(1));
  const int32_t out_channels =
      utils::safe_downcast<int32_t>(weight_sizes.at(0));
  const int32_t kernel_size = utils::safe_downcast<int32_t>(weight_sizes.at(2));
  const int32_t stride_size =
      utils::safe_downcast<int32_t>(graph.get_int_list(stride)->at(0));
  const int32_t padding_size =
      utils::safe_downcast<int32_t>(graph.get_int_list(padding)->at(0));
  const int32_t dilation_size =
      utils::safe_downcast<int32_t>(graph.get_int_list(dilation)->at(0));
  const int32_t in_group_size =
      utils::safe_downcast<int32_t>(in_channels / groups_val);
  const int32_t out_group_size =
      utils::safe_downcast<int32_t>(out_channels / groups_val);

  Kernel1dParams kernel_params = {
      kernel_size,
      stride_size,
      padding_size,
      dilation_size,
      in_group_size,
      out_group_size};

  const OutputParams out_params = {out_min_val, out_max_val};

  std::string kernel_name("conv1d");
  if (clamp_out) {
    kernel_name += "_clamp";
  }
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      conv1d_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in, arg_weight, arg_bias}, vkapi::kRead}},
      // Shader params buffers
      {
          graph.logical_limits_ubo(out),
          graph.sizes_ubo(in),
          graph.create_params_buffer(kernel_params),
          graph.create_params_buffer(out_params),
      },
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(in),
       graph.hashed_layout_of(arg_weight),
       graph.hashed_layout_of(arg_bias)},
      // Resize Args
      {weight, stride, padding, dilation},
      // Resizing Logic
      resize_conv1d_node));
}

} // namespace vkcompute
