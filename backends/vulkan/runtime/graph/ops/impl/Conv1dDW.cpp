/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <limits>

namespace vkcompute {

void resize_conv1d_dw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));

  const int64_t stride = graph->get_int_list(extra_args.at(1))->at(0);
  const int64_t padding = graph->get_int_list(extra_args.at(2))->at(0);
  const int64_t dilation = graph->get_int_list(extra_args.at(3))->at(0);

  const std::vector<int64_t> in_sizes = graph->sizes_of(self);
  const int64_t kernel_size = weight_ref->sizes.at(2);
  const int64_t L_in = in_sizes.at(2);

  const int64_t L_out =
      calc_out_size(L_in, kernel_size, stride, padding, dilation, false);

  graph->virtual_resize(out, {in_sizes.at(0), in_sizes.at(1), L_out});
}

struct Conv1dDWParams final {
  int32_t kernel_size;
  int32_t stride;
  int32_t padding;
  int32_t dilation;
};

struct Conv1dDWClampParams final {
  float output_min;
  float output_max;
};

utils::uvec3 pick_conv1d_dw_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  // out is [N, C, L_out]; in WHCN: {L_out, C, N, 1}
  const uint32_t C = graph->size_at<uint32_t>(-2, out);
  const uint32_t L_out = graph->size_at<uint32_t>(-1, out);
  const uint32_t N =
      graph->dim_of(out) >= 3 ? graph->size_at<uint32_t>(-3, out) : 1;

  return {utils::div_up_4(C), L_out, N};
}

void add_conv1d_dw_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride_ref,
    const ValueRef padding_ref,
    const ValueRef dilation_ref,
    const ValueRef out,
    const float output_min = std::numeric_limits<float>::lowest(),
    const float output_max = std::numeric_limits<float>::max()) {
  VK_CHECK_COND(graph.packed_dim_of(in) == WHCN::kHeightDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kHeightDim);

  const utils::StorageType storage_type = graph.storage_type_of(out);

  // Weight [C, 1, K] prepacked as channels-packed so each vec4 load gives
  // 4 channels at one kernel position.
  ValueRef packed_weight = prepack_standard(
      graph, weight_data, storage_type, utils::kChannelsPacked);

  bool has_bias = graph.val_is_not_none(bias);
  ValueRef packed_bias = kDummyValueRef;
  if (has_bias) {
    packed_bias =
        prepack_standard(graph, bias, storage_type, utils::kWidthPacked);
  }

  const auto stride_val = graph.get_int_list(stride_ref)->at(0);
  const auto padding_val = graph.get_int_list(padding_ref)->at(0);
  const auto dilation_val = graph.get_int_list(dilation_ref)->at(0);

  Conv1dDWParams params{
      utils::safe_downcast<int32_t>(graph.get_tref(weight_data)->sizes.at(2)),
      utils::safe_downcast<int32_t>(stride_val),
      utils::safe_downcast<int32_t>(padding_val),
      utils::safe_downcast<int32_t>(dilation_val),
  };

  Conv1dDWClampParams clamp_params{
      output_min,
      output_max,
  };

  std::string kernel_name = has_bias ? "conv1d_dw_bias" : "conv1d_dw";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, storage_type);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  std::vector<ValueRef> read_inputs = {in, packed_weight};
  if (has_bias) {
    read_inputs.push_back(packed_bias);
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_conv1d_dw_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(in), graph.sizes_ubo(out)},
      // Push Constants
      {PushConstantDataInfo(&params, sizeof(Conv1dDWParams)),
       PushConstantDataInfo(&clamp_params, sizeof(Conv1dDWClampParams))},
      // Specialization Constants
      {},
      // Resize Args
      {weight_data, stride_ref, padding_ref, dilation_ref},
      // Resizing Logic
      resize_conv1d_dw_node));
}

// Args: in, weight, bias, stride, padding, dilation, groups,
//       output_min, output_max, out
// output_min and output_max may be kDummyValueRef (no clamp).
void conv1d_dw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef in = args[0];
  ValueRef weight = args[1];
  ValueRef bias = args[2];
  ValueRef stride = args[3];
  ValueRef padding = args[4];
  ValueRef dilation = args[5];
  ValueRef out = args[9];

  float output_min = std::numeric_limits<float>::lowest();
  float output_max = std::numeric_limits<float>::max();
  if (is_valid(args[7])) {
    output_min = graph.extract_scalar<float>(args[7]);
  }
  if (is_valid(args[8])) {
    output_max = graph.extract_scalar<float>(args[8]);
  }

  add_conv1d_dw_node(
      graph,
      in,
      weight,
      bias,
      stride,
      padding,
      dilation,
      out,
      output_min,
      output_max);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv1d_dw.default, conv1d_dw);
}

} // namespace vkcompute
