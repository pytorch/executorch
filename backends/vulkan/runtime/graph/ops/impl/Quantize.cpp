/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

namespace vkcompute {

namespace {

void resize_quantize_output(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  graph->virtual_resize(out, graph->sizes_of(in));
}

} // namespace

void add_quantize_per_tensor_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("quantize_per_tensor");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  float scale_val = static_cast<float>(graph.get_double(scale));
  int zero_point_val = static_cast<int>(graph.get_int(zero_point));
  int quant_min_val = static_cast<int>(graph.get_int(quant_min));
  int quant_max_val = static_cast<int>(graph.get_int(quant_max));

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.numel_ubo(input),
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output)};
    push_constants = {
        PushConstantDataInfo(&scale_val, sizeof(float)),
        PushConstantDataInfo(&zero_point_val, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input), graph.logical_limits_ubo(output)};
    push_constants = {
        PushConstantDataInfo(&scale_val, sizeof(float)),
        PushConstantDataInfo(&zero_point_val, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  }

  vkapi::SpecVarList spec_vars = {
      graph.hashed_layout_of(output),
      graph.hashed_layout_of(input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {},
      // Resizing Logic
      resize_quantize_output));
}

void add_quantize_per_token_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("quantize_per_token");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  int quant_min_val = static_cast<int>(graph.get_int(quant_min));
  int quant_max_val = static_cast<int>(graph.get_int(quant_max));

  int num_tokens = static_cast<int>(graph.sizes_of(scale)[0]);

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.numel_ubo(input),
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output),
    };
    push_constants = {
        PushConstantDataInfo(&num_tokens, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(output),
    };
    push_constants = {
        PushConstantDataInfo(&num_tokens, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  }

  vkapi::SpecVarList spec_vars = {
      graph.hashed_layout_of(output),
      graph.hashed_layout_of(input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {input, vkapi::kRead},
       {{scale, zero_point}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {},
      // Resizing Logic
      resize_quantize_output));
}

void quantize_per_tensor_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(output));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kDouble ||
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf);

  add_quantize_per_tensor_node(
      graph, input, scale, zero_point, quant_min, quant_max, output);
}

void quantize_per_token_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale));
  VK_CHECK_COND(graph.val_is_tensor(zero_point));
  VK_CHECK_COND(graph.val_is_tensor(output));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kDouble ||
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf);

  // Check that scale and zero_point have buffer storage and width packing
  VK_CHECK_COND(graph.is_buffer_storage(scale));
  VK_CHECK_COND(graph.packed_dim_of(scale) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.is_buffer_storage(zero_point));
  VK_CHECK_COND(graph.packed_dim_of(zero_point) == WHCN::kWidthDim);

  // Check that tensors with texture storage have standard axis map
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.has_standard_axis_map(input));
  }
  if (!graph.is_buffer_storage(output)) {
    VK_CHECK_COND(graph.has_standard_axis_map(output));
  }

  // Calculate number of tokens (product of all dimensions except the last one)
  int64_t num_tokens = 1;
  const auto input_sizes = graph.sizes_of(input);
  for (size_t i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  const auto scale_sizes = graph.sizes_of(scale);
  const auto zero_point_sizes = graph.sizes_of(zero_point);

  VK_CHECK_COND(scale_sizes.size() == 1);
  VK_CHECK_COND(zero_point_sizes.size() == 1);
  VK_CHECK_COND(scale_sizes[0] == num_tokens);
  VK_CHECK_COND(zero_point_sizes[0] == num_tokens);

  add_quantize_per_token_node(
      graph, input, scale, zero_point, quant_min, quant_max, output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(quantize_per_tensor.default, quantize_per_tensor_impl);
  VK_REGISTER_OP(quantize_per_token.default, quantize_per_token_impl);
}

} // namespace vkcompute
