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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

namespace vkcompute {

namespace {

void resize_choose_qparams_tensor_output(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef scale_out = args.at(0).refs.at(0);
  const ValueRef zero_point_out = args.at(0).refs.at(1);

  // Both scale and zero_point are scalar tensors for per-tensor quantization
  // Since we use single workgroup approach, no extra buffer space needed
  graph->virtual_resize(scale_out, {});
  graph->virtual_resize(zero_point_out, {});
}

void resize_choose_qparams_per_token_output(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef scale_out = args.at(0).refs.at(0);
  const ValueRef zero_point_out = args.at(0).refs.at(1);
  const ValueRef input = args.at(1).refs.at(0);

  // Calculate output sizes for scale and zero_point tensors
  const auto input_sizes = graph->sizes_of(input);
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(input_sizes.size() - 1);
  for (size_t i = 0; i < input_sizes.size() - 1; i++) {
    output_sizes.push_back(input_sizes[i]);
  }
  output_sizes.push_back(1);

  graph->virtual_resize(scale_out, output_sizes);
  graph->virtual_resize(zero_point_out, output_sizes);
}

// Custom workgroup size pickers for ChooseQParams operations
utils::uvec3 choose_qparams_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  // For per-tensor quantization, we want a single workgroup that can handle
  // all elements with proper reduction. The shader uses NWORKERS=64 threads.
  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For buffer storage, use a single workgroup in X dimension
    // The shader will handle strided access across all elements
    return {1u, 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_global_wg_size(args.at(0).refs.at(0));
  }
}

utils::uvec3 choose_qparams_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For buffer storage, use 64 threads in X dimension to match NWORKERS
    // This ensures the shared memory arrays are properly sized
    return {64u, 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_local_wg_size(global_workgroup_size);
  }
}

utils::uvec3 choose_qparams_per_token_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For per-token quantization, we need one workgroup per token
    // Calculate number of tokens (product of all dimensions except the last
    // one)
    const auto input_sizes = graph->sizes_of(input);
    int64_t num_tokens = 1;
    for (size_t i = 0; i < input_sizes.size() - 1; i++) {
      num_tokens *= input_sizes[i];
    }

    return {static_cast<uint32_t>(num_tokens), 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_global_wg_size(args.at(0).refs.at(0));
  }
}

utils::uvec3 choose_qparams_per_token_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For buffer storage, use 64 threads in X dimension to match NWORKERS
    return {64u, 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_local_wg_size(global_workgroup_size);
  }
}

} // namespace

void add_choose_qparams_tensor_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& scale_out,
    const ValueRef& zero_point_out) {
  std::string kernel_name("choose_qparams_tensor");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));

  int quant_min_val = static_cast<int>(graph.get_int(quant_min));
  int quant_max_val = static_cast<int>(graph.get_int(quant_max));

  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(scale_out),
        graph.strides_ubo(scale_out),
        graph.sizes_ubo(zero_point_out),
        graph.strides_ubo(zero_point_out)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(scale_out),
        graph.logical_limits_ubo(zero_point_out)};
  }

  std::vector<PushConstantDataInfo> push_constants;
  push_constants = {
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      choose_qparams_pick_global_wg_size,
      choose_qparams_pick_local_wg_size,
      // Inputs and Outputs
      {{scale_out, vkapi::kWrite},
       {zero_point_out, vkapi::kWrite},
       {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_choose_qparams_tensor_output));
}

void add_choose_qparams_per_token_asymmetric_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale_out,
    const ValueRef& zero_point_out) {
  std::string kernel_name("choose_qparams_per_token_asymmetric");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));

  // Calculate number of tokens (product of all dimensions except the last one)
  int64_t num_tokens = 1;
  const auto input_sizes = graph.sizes_of(input);
  for (size_t i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  int num_tokens_val = static_cast<int>(num_tokens);
  int quant_min_val = -128; // Fixed for asymmetric quantization
  int quant_max_val = 127; // Fixed for asymmetric quantization

  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(scale_out),
        graph.strides_ubo(scale_out),
        graph.sizes_ubo(zero_point_out),
        graph.strides_ubo(zero_point_out)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(scale_out),
        graph.logical_limits_ubo(zero_point_out)};
  }

  std::vector<PushConstantDataInfo> push_constants;
  push_constants = {
      PushConstantDataInfo(&num_tokens_val, sizeof(int)),
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      choose_qparams_per_token_pick_global_wg_size,
      choose_qparams_per_token_pick_local_wg_size,
      // Inputs and Outputs
      {{scale_out, vkapi::kWrite},
       {zero_point_out, vkapi::kWrite},
       {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_choose_qparams_per_token_output));
}

void choose_qparams_tensor_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef scale_out = args[arg_idx++];
  const ValueRef zero_point_out = args[arg_idx++];

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale_out));
  VK_CHECK_COND(graph.val_is_tensor(zero_point_out));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf ||
      graph.dtype_of(input) == vkapi::kDouble);

  // Verify output types - accept CPU types but convert to GPU types
  VK_CHECK_COND(
      graph.dtype_of(scale_out) == vkapi::kFloat ||
      graph.dtype_of(scale_out) == vkapi::kDouble);
  VK_CHECK_COND(
      graph.dtype_of(zero_point_out) == vkapi::kInt ||
      graph.dtype_of(zero_point_out) == vkapi::kLong);

  // Check that texture storage is width packed
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  }

  add_choose_qparams_tensor_node(
      graph, input, quant_min, quant_max, scale_out, zero_point_out);
}

void choose_qparams_per_token_asymmetric_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale_out = args[arg_idx++];
  const ValueRef zero_point_out = args[arg_idx++];

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale_out));
  VK_CHECK_COND(graph.val_is_tensor(zero_point_out));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf ||
      graph.dtype_of(input) == vkapi::kDouble);

  // Verify output types - accept CPU types but convert to GPU types
  VK_CHECK_COND(
      graph.dtype_of(scale_out) == vkapi::kFloat ||
      graph.dtype_of(scale_out) == vkapi::kDouble);
  VK_CHECK_COND(
      graph.dtype_of(zero_point_out) == vkapi::kInt ||
      graph.dtype_of(zero_point_out) == vkapi::kLong);

  add_choose_qparams_per_token_asymmetric_node(
      graph, input, scale_out, zero_point_out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(choose_qparams.tensor, choose_qparams_tensor_impl);
  VK_REGISTER_OP(
      choose_qparams_per_token_asymmetric.default,
      choose_qparams_per_token_asymmetric_impl);
}

} // namespace vkcompute
