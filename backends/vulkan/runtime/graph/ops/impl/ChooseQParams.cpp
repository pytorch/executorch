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

void resize_choose_qparams_tensor_output(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr scale_out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr zero_point_out = graph->get_tensor(args[0].refs[1]);

  // Both scale and zero_point are scalar tensors for per-tensor quantization
  // Since we use single workgroup approach, no extra buffer space needed
  scale_out->virtual_resize({});
  zero_point_out->virtual_resize({});
}

void resize_choose_qparams_per_token_output(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr scale_out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr zero_point_out = graph->get_tensor(args[0].refs[1]);
  vTensorPtr input = graph->get_tensor(args[1].refs[0]);

  // Calculate output sizes for scale and zero_point tensors
  std::vector<int64_t> output_sizes;
  for (size_t i = 0; i < input->sizes().size() - 1; i++) {
    output_sizes.push_back(input->sizes()[i]);
  }
  output_sizes.push_back(1);

  scale_out->virtual_resize(output_sizes);
  zero_point_out->virtual_resize(output_sizes);
}

utils::uvec3 choose_qparams_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  // For global reduction, we need to process the entire input tensor
  const ValueRef input = args.at(0).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    const uint32_t local_threads = 64; // From choose_qparams_local_wg_size

    // For per-tensor quantization, use SINGLE WORKGROUP approach to avoid
    // complex multi-workgroup synchronization issues that cause race
    // conditions. A single workgroup with 64 threads can efficiently process
    // large tensors by having each thread process multiple elements with
    // stride.

    // Return single workgroup with 64 threads
    return {local_threads, 1u, 1u};
  } else {
    // For texture storage, use single workgroup approach for reliability
    const uint32_t local_threads = 64; // From choose_qparams_local_wg_size

    // Return single workgroup with 64 threads
    return {local_threads, 1u, 1u};
  }
}

utils::uvec3 choose_qparams_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  (void)global_workgroup_size;

  const ValueRef input = args.at(0).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For hierarchical reduction, use 64 threads per work group for better
    // efficiency This provides better GPU utilization while still being
    // manageable for shared memory

    const uint32_t local_threads = 64;
    return {local_threads, 1u, 1u};
  } else {
    // For texture storage, use default local workgroup size
    return graph->create_local_wg_size(global_workgroup_size);
  }
}

utils::uvec3 choose_qparams_per_token_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(0).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For per-token reduction, we need one workgroup per token
    // Calculate number of tokens (product of all dimensions except the last
    // one)
    int64_t num_tokens = 1;
    const auto input_sizes = graph->sizes_of(input);
    for (size_t i = 0; i < input_sizes.size() - 1; i++) {
      num_tokens *= input_sizes[i];
    }

    // GPU hardware limits: Most GPUs support max ~65535 workgroups per
    // dimension
    const uint32_t max_workgroups = 65535;
    const uint32_t local_x = 64u; // From choose_qparams_per_token_local_wg_size

    // Clamp number of workgroups to hardware limits
    uint32_t clamped_workgroups =
        std::min(static_cast<uint32_t>(num_tokens), max_workgroups);

    // If we have more tokens than workgroups, each workgroup will process
    // multiple tokens

    // Calculate total threads needed
    const uint32_t total_threads_x = clamped_workgroups * local_x;
    const uint32_t total_threads_y = 1u;
    const uint32_t total_threads_z = 1u;

    return {total_threads_x, total_threads_y, total_threads_z};
  } else {
    // For texture storage, calculate number of tokens
    int64_t num_tokens = 1;
    const auto input_sizes = graph->sizes_of(input);
    for (size_t i = 0; i < input_sizes.size() - 1; i++) {
      num_tokens *= input_sizes[i];
    }

    // For texture storage, clamp to reasonable limits for performance
    // Large token counts (>1024) can cause very slow execution
    const uint32_t max_reasonable_tokens = 1024;
    const uint32_t local_x = 64u; // From choose_qparams_per_token_local_wg_size

    uint32_t clamped_workgroups =
        std::min(static_cast<uint32_t>(num_tokens), max_reasonable_tokens);

    // Calculate total threads needed
    const uint32_t total_threads_x = clamped_workgroups * local_x;
    const uint32_t total_threads_y = 1u;
    const uint32_t total_threads_z = 1u;

    return {total_threads_x, total_threads_y, total_threads_z};
  }
}

utils::uvec3 choose_qparams_per_token_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  (void)global_workgroup_size;

  const ValueRef input = args.at(0).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For per-token reduction, each workgroup processes one token
    // Use 64 threads per work group to match shared memory allocation
    const uint32_t local_threads = 64;

    return {local_threads, 1u, 1u};
  } else {
    // For texture storage, use default local workgroup size
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
      choose_qparams_global_wg_size,
      choose_qparams_local_wg_size,
      // Inputs and Outputs
      {{input, vkapi::kRead},
       {scale_out, vkapi::kWrite},
       {zero_point_out, vkapi::kWrite}},
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
      choose_qparams_per_token_global_wg_size,
      choose_qparams_per_token_local_wg_size,
      // Inputs and Outputs
      {{input, vkapi::kRead},
       {scale_out, vkapi::kWrite},
       {zero_point_out, vkapi::kWrite}},
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
