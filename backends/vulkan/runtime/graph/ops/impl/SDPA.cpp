/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/RepeatInterleave.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Slice.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Softmax.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Transpose.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_kv_cache_update_node(
    ComputeGraph& graph,
    const ValueRef input_pos_symint,
    const ValueRef projected,
    const ValueRef cache) {
  std::string kernel_name("kv_cache_update");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(projected));
  add_dtype_suffix(kernel_name, graph.dtype_of(projected));

  utils::uvec3 global_size;
  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(cache)) {
    global_size = graph.create_global_wg_size(projected);

    param_ubos = {
        graph.numel_ubo(projected),
        graph.strides_ubo(cache),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    global_size = graph.logical_limits_of(projected);

    param_ubos = {
        graph.logical_limits_ubo(projected),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  }
  const utils::uvec3 local_size = graph.create_local_wg_size(global_size);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{cache, vkapi::kWrite}, {projected, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Specialization Constants
      {},
      // Resizing Logic
      nullptr,
      {}));
}

void add_attn_weight_scale_and_mask_node(
    ComputeGraph& graph,
    const ValueRef input_pos_symint,
    const ValueRef q_projected,
    const ValueRef attn_weight) {
  std::string kernel_name("sdpa_attn_weight_scale_and_mask");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(attn_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(attn_weight));

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  utils::uvec3 global_size;
  utils::uvec3 local_size;
  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(attn_weight)) {
    global_size = {
        graph.size_at<uint32_t>(-1, attn_weight),
        graph.size_at<uint32_t>(-2, attn_weight),
        graph.size_at<uint32_t>(-3, attn_weight),
    };

    param_ubos = {
        graph.sizes_ubo(attn_weight),
        graph.strides_ubo(attn_weight),
        graph.create_params_buffer(scale_val)};
  } else {
    global_size = graph.logical_limits_of(attn_weight);

    param_ubos = {
        graph.logical_limits_ubo(attn_weight),
        graph.get_or_create_int_param_buffer(input_pos_symint),
        graph.create_params_buffer(scale_val)};
  }

  local_size = graph.create_local_wg_size(global_size);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{attn_weight, vkapi::kReadWrite}},
      // Shader param buffers
      param_ubos,
      // Specialization Constants
      {},
      // Resizing Logic
      nullptr,
      {}));
}

std::vector<int64_t> get_cache_slice_sizes(
    ComputeGraph& graph,
    ValueRef cache,
    ValueRef input_pos_symint,
    ValueRef q_projected) {
  std::vector<int64_t> slice_sizes = graph.sizes_of(cache);

  // Cache slicing will always be in the channels dim
  const int32_t input_pos_val = graph.read_symint(input_pos_symint);
  const int64_t q_seq_len = graph.size_at<int64_t>(1, q_projected);
  slice_sizes.at(1) = input_pos_val + q_seq_len;
  return slice_sizes;
}

void resize_cache_slice_view_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;
  std::vector<int64_t> slice_sizes = get_cache_slice_sizes(
      *graph, extra_args[0], extra_args[1], extra_args[2]);

  graph->get_tensor(extra_args[3])->virtual_resize(slice_sizes);
}

void add_cache_slice_view_node(
    ComputeGraph& graph,
    ValueRef cache,
    ValueRef input_pos_symint,
    ValueRef q_projected,
    ValueRef cache_sliced,
    const int64_t max_seq_len) {
  std::vector<int64_t> slice_sizes =
      get_cache_slice_sizes(graph, cache, input_pos_symint, q_projected);
  // Initialize the slice to the maximum possible size to start
  slice_sizes.at(1) = max_seq_len;

  graph.get_tensor(cache_sliced)->virtual_resize(slice_sizes);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      resize_cache_slice_view_node,
      {cache, input_pos_symint, q_projected, cache_sliced}));
}

void resize_sdpa_out(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;

  int arg_idx = 0;
  const ValueRef q_projected = extra_args[arg_idx++];
  const ValueRef out = extra_args[arg_idx++];
  graph->get_tensor(out)->virtual_resize(graph->sizes_of(q_projected));
}

void sdpa_with_kv_cache_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_projected = args[arg_idx++];
  const ValueRef v_projected = args[arg_idx++];
  const ValueRef k_cache_data = args[arg_idx++];
  const ValueRef v_cache_data = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef sequence_len = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef dropout_p = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];

  // Output tensors
  const ValueRef out = args[arg_idx++];

  // Unused variables
  (void)sequence_len;

  // Batches must be 1
  VK_CHECK_COND(graph.size_at<int32_t>(-4, q_projected) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, k_projected) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, v_projected) == 1);
  // k and v projected must have the same shape
  VK_CHECK_COND(graph.sizes_of(k_projected) == graph.sizes_of(v_projected));
  // head dim must match between tensors
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, q_projected) ==
      graph.size_at<int32_t>(-1, k_projected));
  // All tensors must have the packed dim be the width (head) dimension
  VK_CHECK_COND(graph.packed_dim_of(q_projected) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k_projected) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v_projected) == WHCN::kWidthDim);
  // Some variables are not supported yet
  VK_CHECK_COND(
      graph.val_is_none(dropout_p) ||
      graph.extract_scalar<double>(dropout_p) == 0);
  VK_CHECK_COND(graph.val_is_none(scale));
  // is_causal is assumed to be true in the current implementation.
  VK_CHECK_COND(
      graph.val_is_none(is_causal) || graph.extract_scalar<bool>(is_causal));
  VK_CHECK_COND(graph.val_is_none(attn_mask));

  const ValueRef k_cache =
      prepack_standard_like(graph, k_cache_data, q_projected);
  const ValueRef v_cache =
      prepack_standard_like(graph, v_cache_data, q_projected);

  const int32_t max_seq_len = graph.size_at<int32_t>(1, k_cache);

  add_kv_cache_update_node(graph, input_pos_symint, k_projected, k_cache);
  add_kv_cache_update_node(graph, input_pos_symint, v_projected, v_cache);

  // Slice caches from 0 to input_pos + sequence_len
  const ValueRef k_cache_sliced = graph.add_tensor_view(k_cache);
  const ValueRef v_cache_sliced = graph.add_tensor_view(v_cache);
  add_cache_slice_view_node(
      graph,
      k_cache,
      input_pos_symint,
      q_projected,
      k_cache_sliced,
      max_seq_len);
  add_cache_slice_view_node(
      graph,
      v_cache,
      input_pos_symint,
      q_projected,
      v_cache_sliced,
      max_seq_len);

  // Scalar values for various dims
  const ValueRef channels = graph.add_scalar<int64_t>(1);
  const ValueRef height = graph.add_scalar<int64_t>(2);
  const ValueRef width = graph.add_scalar<int64_t>(3);

  // Repeat interleave
  const int64_t num_heads = graph.size_at<int64_t>(2, q_projected);
  const int64_t num_kv_heads = graph.size_at<int64_t>(2, k_projected);

  const ValueRef num_repeats =
      graph.add_scalar<int64_t>(num_heads / num_kv_heads);

  std::vector<int64_t> cache_slice_repeated_sizes(graph.sizes_of(q_projected));
  cache_slice_repeated_sizes.at(1) = max_seq_len;

  TmpTensor k_cache_sliced_repeated(
      &graph, cache_slice_repeated_sizes, graph.dtype_of(k_cache_sliced));
  TmpTensor v_cache_sliced_repeated(
      &graph, cache_slice_repeated_sizes, graph.dtype_of(v_cache_sliced));

  add_repeat_interleave_node(
      graph, k_cache_sliced, num_repeats, height, k_cache_sliced_repeated);
  add_repeat_interleave_node(
      graph, v_cache_sliced, num_repeats, height, v_cache_sliced_repeated);

  // Transpose sequence and head dims
  const ValueRef q_transposed = graph.add_tensor_view(q_projected);
  const ValueRef k_transposed = graph.add_tensor_view(k_cache_sliced_repeated);
  const ValueRef v_transposed = graph.add_tensor_view(v_cache_sliced_repeated);

  add_transpose_view_node(graph, q_projected, channels, height, q_transposed);
  add_transpose_view_node(
      graph, k_cache_sliced_repeated, channels, height, k_transposed);
  add_transpose_view_node(
      graph, v_cache_sliced_repeated, channels, height, v_transposed);

  // Transpose K again to prepare for matmul
  const ValueRef k_transposed_2 = graph.add_tensor_view(k_transposed);
  add_transpose_view_node(graph, k_transposed, height, width, k_transposed_2);

  // Initialize attn_weight to the maximum possible size
  std::vector<int64_t> attn_weight_full_sizes = graph.sizes_of(q_transposed);
  attn_weight_full_sizes.at(2) = max_seq_len;
  attn_weight_full_sizes.at(3) = max_seq_len;
  TmpTensor attn_weight(
      &graph, attn_weight_full_sizes, graph.dtype_of(q_transposed));

  // Resize attn_weight to the correct dim
  std::vector<int64_t> attn_weight_sizes = attn_weight_full_sizes;
  attn_weight_sizes.at(2) = graph.size_at<int64_t>(2, q_transposed);
  attn_weight_sizes.at(3) = graph.size_at<int64_t>(2, k_transposed);
  graph.get_tensor(attn_weight)->virtual_resize(attn_weight_sizes);

  // Calculate attention weight, which is a matmul of Q and K
  const ValueRef mat2_is_transposed = graph.add_scalar<bool>(false);
  add_matmul_node(
      graph, q_transposed, k_transposed_2, attn_weight, mat2_is_transposed);

  // Apply scale and mask to the attention weight
  add_attn_weight_scale_and_mask_node(
      graph, input_pos_symint, q_projected, attn_weight);

  TmpTensor attn_weight_softmax(
      &graph, attn_weight_full_sizes, graph.dtype_of(q_transposed));
  graph.get_tensor(attn_weight_softmax)->virtual_resize(attn_weight_sizes);
  add_softmax_node(graph, attn_weight, width, attn_weight_softmax, false);

  // Calculate final output
  const ValueRef out_transposed = graph.add_tensor_view(out);
  add_transpose_view_node(graph, out, channels, height, out_transposed);
  add_matmul_node(
      graph,
      attn_weight_softmax,
      v_transposed,
      out_transposed,
      mat2_is_transposed);

  graph.execute_nodes().emplace_back(
      new ExecuteNode(resize_sdpa_out, {q_projected, out}));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
}

} // namespace vkcompute
