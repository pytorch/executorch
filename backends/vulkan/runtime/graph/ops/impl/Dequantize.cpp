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

namespace vkcompute {

namespace {

void resize_dequantize_output(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr input = graph->get_tensor(args[1].refs[0]);
  out->virtual_resize(input->sizes());
}

} // namespace

void add_dequantize_per_tensor_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("dequantize_per_tensor");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  float scale_val = static_cast<float>(graph.get_double(scale));
  int zero_point_val = static_cast<int>(graph.get_int(zero_point));
  int quant_min_val = static_cast<int>(graph.get_int(quant_min));
  int quant_max_val = static_cast<int>(graph.get_int(quant_max));

  utils::uvec3 global_size;
  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(input)) {
    global_size = graph.create_global_wg_size(input);

    param_ubos = {
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output)};
  } else {
    global_size = graph.logical_limits_of(input);

    param_ubos = {
        graph.logical_limits_ubo(input), graph.logical_limits_ubo(output)};
  }

  const utils::uvec3 local_size = graph.create_local_wg_size(global_size);

  std::vector<PushConstantDataInfo> push_constants;
  push_constants = {
      PushConstantDataInfo(&scale_val, sizeof(float)),
      PushConstantDataInfo(&zero_point_val, sizeof(int)),
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
  };

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{input, vkapi::kRead}, {output, vkapi::kReadWrite}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_dequantize_output));
}

void add_dequantize_per_token_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("dequantize_per_token");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  int quant_min_val = static_cast<int>(graph.get_int(quant_min));
  int quant_max_val = static_cast<int>(graph.get_int(quant_max));

  int num_tokens = static_cast<int>(graph.sizes_of(scale)[0]);

  utils::uvec3 global_size;
  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(input)) {
    global_size = graph.create_global_wg_size(input);

    param_ubos = {
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output),
    };
  } else {
    global_size = graph.logical_limits_of(input);

    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(output),
    };
  }

  const utils::uvec3 local_size = graph.create_local_wg_size(global_size);

  std::vector<PushConstantDataInfo> push_constants;
  push_constants = {
      PushConstantDataInfo(&num_tokens, sizeof(int)),
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
  };

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{input, vkapi::kRead},
       {output, vkapi::kWrite},
       {{scale, zero_point}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_dequantize_output));
}

void dequantize_per_tensor_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Verify input is an integer type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kByte ||
      graph.dtype_of(input) == vkapi::kChar ||
      graph.dtype_of(input) == vkapi::kShort ||
      graph.dtype_of(input) == vkapi::kInt);

  // Verify output is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(output) == vkapi::kFloat ||
      graph.dtype_of(output) == vkapi::kDouble);

  // Resize output tensor to match input tensor shape
  graph.get_tensor(output)->virtual_resize(graph.sizes_of(input));

  add_dequantize_per_tensor_node(
      graph, input, scale, zero_point, quant_min, quant_max, output);
}

void dequantize_per_token_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Verify input is an integer type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kByte ||
      graph.dtype_of(input) == vkapi::kChar ||
      graph.dtype_of(input) == vkapi::kShort ||
      graph.dtype_of(input) == vkapi::kInt);

  // Verify output is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(output) == vkapi::kFloat ||
      graph.dtype_of(output) == vkapi::kDouble);

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

  // Resize output tensor to match input tensor shape
  graph.get_tensor(output)->virtual_resize(graph.sizes_of(input));

  add_dequantize_per_token_node(
      graph, input, scale, zero_point, quant_min, quant_max, output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(dequantize_per_tensor.default, dequantize_per_tensor_impl);
  VK_REGISTER_OP(dequantize_per_token.default, dequantize_per_token_impl);
}

} // namespace vkcompute
