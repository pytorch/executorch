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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/RepeatInterleave.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Slice.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Softmax.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Transpose.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_sdpa_out(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;

  int arg_idx = 0;
  const ValueRef q_projected = extra_args[arg_idx++];
  const ValueRef out = extra_args[arg_idx++];
  graph->virtual_resize(out, graph->sizes_of(q_projected));
}

void resize_flash_attention_out(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  // Find the output tensor in the args - it's the first tensor in the first
  // ArgGroup
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef q_projected = args.at(1).refs.at(0);
  graph->virtual_resize(out, graph->sizes_of(q_projected));
}

utils::uvec3 flash_attention_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;

  const ValueRef q_projected = resize_args.at(0);
  const ValueRef block_size_r = resize_args.at(1);

  // Get tensor dimensions - PyTorch format is [B, N, H, D]
  // But Vulkan uses negative indexing: -4=B, -3=N, -2=H, -1=D
  const int32_t B = graph->size_at<int32_t>(-4, q_projected); // batch
  const int32_t N = graph->size_at<int32_t>(-3, q_projected); // sequence length
  const int32_t H = graph->size_at<int32_t>(-2, q_projected); // num heads
  const int32_t Br =
      static_cast<int32_t>(graph->extract_scalar<int64_t>(block_size_r));

  // Calculate number of row blocks
  const int32_t Tr = (N + Br - 1) / Br;

  return {static_cast<uint32_t>(B * H * Tr), 1, 1};
}

void flash_attention_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_cache = args[arg_idx++];
  const ValueRef v_cache = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef dropout_p = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];

  const ValueRef out = args[arg_idx++];

  // Extract input_pos value for causal masking
  const int32_t input_pos_val = graph.read_symint(input_pos_symint);

  const ValueRef k_cache_tensor = k_cache;
  const ValueRef v_cache_tensor = v_cache;

  // Validation checks - re-enable with correct indexing
  VK_CHECK_COND(graph.size_at<int32_t>(-4, q_projected) == 1); // batch size = 1
  VK_CHECK_COND(graph.size_at<int32_t>(-4, k_cache_tensor) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, v_cache_tensor) == 1);
  VK_CHECK_COND(
      graph.sizes_of(k_cache_tensor) == graph.sizes_of(v_cache_tensor));
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, q_projected) ==
      graph.size_at<int32_t>(-1, k_cache_tensor)); // head_dim must match
  VK_CHECK_COND(
      graph.val_is_none(dropout_p) ||
      graph.extract_scalar<double>(dropout_p) == 0);
  VK_CHECK_COND(graph.val_is_none(scale));
  VK_CHECK_COND(
      graph.val_is_none(is_causal) || graph.extract_scalar<bool>(is_causal));
  VK_CHECK_COND(graph.val_is_none(attn_mask));

  if (graph.is_buffer_storage(q_projected)) {
    VK_CHECK_COND(graph.is_buffer_storage(k_cache_tensor));
    VK_CHECK_COND(graph.is_buffer_storage(v_cache_tensor));
    VK_CHECK_COND(graph.is_buffer_storage(out));
  }

  // Calculate scale factor
  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  // Get number of heads for multi-query attention support
  const int32_t num_heads = graph.size_at<int32_t>(-2, q_projected);
  const int32_t num_kv_heads = graph.size_at<int32_t>(-2, k_cache_tensor);

  const int32_t block_size_r = 32; // Row block size
  const int32_t block_size_c = 32; // Column block size

  // l and m have shape [B, H, N]
  std::vector<int64_t> lm_sizes = {
      graph.size_at<int64_t>(-4, q_projected), // B (batch)
      graph.size_at<int64_t>(-2, q_projected), // H (num heads)
      graph.size_at<int64_t>(-3, q_projected) // N (sequence length)
  };

  // t_l stores row-wise normalization sums for softmax computation
  // t_m stores row-wise maximum values for numerical stability in softmax
  TmpTensor t_l(&graph, lm_sizes, vkapi::kFloat, graph.storage_type_of(out));
  TmpTensor t_m(&graph, lm_sizes, vkapi::kFloat, graph.storage_type_of(out));

  // Choose kernel name based on storage type
  std::string kernel_name = "flash_attention";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected), // Q_sizes
      graph.sizes_ubo(k_cache_tensor), // K_sizes
      graph.sizes_ubo(v_cache_tensor), // V_sizes
      graph.sizes_ubo(out), // O_sizes
      graph.sizes_ubo(t_l), // l_sizes
      graph.sizes_ubo(t_m), // m_sizes
      graph.create_params_buffer(scale_val), // scale
      graph.create_params_buffer(block_size_r), // block_size_r
      graph.create_params_buffer(block_size_c), // block_size_c
      graph.create_params_buffer(input_pos_val), // input_pos
      graph.create_params_buffer(num_heads), // num_heads
      graph.create_params_buffer(num_kv_heads) // num_kv_heads
  };

  // Create block size references for dispatch calculation
  const ValueRef block_size_r_ref =
      graph.add_scalar<int64_t>(static_cast<int64_t>(block_size_r));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      flash_attention_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {
          {{out, t_l, t_m}, vkapi::kReadWrite},
          {{q_projected, k_cache_tensor, v_cache_tensor}, vkapi::kRead},
      },
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {q_projected, block_size_r_ref},
      // Resizing Logic
      resize_flash_attention_out));
}

utils::uvec3 kv_cache_update_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef cache = args.at(0).refs.at(0);
  const ValueRef projected = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(cache)) {
    return graph->create_global_wg_size(projected);
  } else {
    return graph->logical_limits_of(projected);
  }
}

void add_kv_cache_update_node(
    ComputeGraph& graph,
    const ValueRef input_pos_symint,
    const ValueRef projected,
    const ValueRef cache) {
  std::string kernel_name("kv_cache_update");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(projected));
  add_dtype_suffix(kernel_name, graph.dtype_of(projected));

  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(cache)) {
    param_ubos = {
        graph.numel_ubo(projected),
        graph.strides_ubo(cache),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(projected),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      kv_cache_update_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{cache, vkapi::kWrite}, {projected, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

utils::uvec3 attn_weight_scale_and_mask_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef attn_weight = args.at(0).refs.at(0);

  if (graph->is_buffer_storage(attn_weight)) {
    return {
        graph->size_at<uint32_t>(-1, attn_weight),
        graph->size_at<uint32_t>(-2, attn_weight),
        graph->size_at<uint32_t>(-3, attn_weight),
    };
  } else {
    return graph->logical_limits_of(attn_weight);
  }
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

  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(attn_weight)) {
    param_ubos = {
        graph.sizes_ubo(attn_weight),
        graph.strides_ubo(attn_weight),
        graph.create_params_buffer(scale_val)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(attn_weight),
        graph.get_or_create_int_param_buffer(input_pos_symint),
        graph.create_params_buffer(scale_val)};
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      attn_weight_scale_and_mask_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{attn_weight, vkapi::kReadWrite}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
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

  graph->virtual_resize(extra_args[3], slice_sizes);
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

  graph.virtual_resize(cache_sliced, slice_sizes);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      resize_cache_slice_view_node,
      {cache, input_pos_symint, q_projected, cache_sliced}));
}

void update_cache_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef value = args[arg_idx++];
  const ValueRef cache = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef out = args[arg_idx++];

  // Unused variables
  (void)out;

  VK_CHECK_COND(graph.size_at<int32_t>(-4, value) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, cache) == 1);
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, value) == graph.size_at<int32_t>(-1, cache));
  VK_CHECK_COND(
      graph.size_at<int32_t>(-2, value) == graph.size_at<int32_t>(-2, cache));

  add_kv_cache_update_node(graph, input_pos_symint, value, cache);
}

void sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q_projected = args[arg_idx++];
  const ValueRef k_cache = args[arg_idx++];
  const ValueRef v_cache = args[arg_idx++];
  const ValueRef input_pos_symint = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef dropout_p = args[arg_idx++];
  const ValueRef is_causal = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];

  // Output tensors
  const ValueRef out = args[arg_idx++];

  // Batches must be 1
  VK_CHECK_COND(graph.size_at<int32_t>(-4, q_projected) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, k_cache) == 1);
  VK_CHECK_COND(graph.size_at<int32_t>(-4, v_cache) == 1);
  // k and v projected must have the same shape
  VK_CHECK_COND(graph.sizes_of(k_cache) == graph.sizes_of(v_cache));
  // head dim must match between tensors
  VK_CHECK_COND(
      graph.size_at<int32_t>(-1, q_projected) ==
      graph.size_at<int32_t>(-1, k_cache));
  // All tensors must have the packed dim be the width (head) dimension
  VK_CHECK_COND(graph.packed_dim_of(q_projected) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k_cache) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v_cache) == WHCN::kWidthDim);
  // Some variables are not supported yet
  VK_CHECK_COND(
      graph.val_is_none(dropout_p) ||
      graph.extract_scalar<double>(dropout_p) == 0);
  VK_CHECK_COND(graph.val_is_none(scale));
  // is_causal is assumed to be true in the current implementation.
  VK_CHECK_COND(
      graph.val_is_none(is_causal) || graph.extract_scalar<bool>(is_causal));
  VK_CHECK_COND(graph.val_is_none(attn_mask));

  const int32_t max_seq_len = graph.size_at<int32_t>(1, k_cache);

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
  const int64_t num_kv_heads = graph.size_at<int64_t>(2, k_cache);

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
  graph.virtual_resize(attn_weight, attn_weight_sizes);

  // Calculate attention weight, which is a matmul of Q and K
  const ValueRef mat2_is_transposed = graph.add_scalar<bool>(false);
  add_matmul_node(
      graph, q_transposed, k_transposed_2, attn_weight, mat2_is_transposed);

  // Apply scale and mask to the attention weight
  add_attn_weight_scale_and_mask_node(
      graph, input_pos_symint, q_projected, attn_weight);

  TmpTensor attn_weight_softmax(
      &graph, attn_weight_full_sizes, graph.dtype_of(q_transposed));
  graph.virtual_resize(attn_weight_softmax, attn_weight_sizes);
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

  (void)sequence_len;

  const ValueRef k_cache =
      prepack_standard_like(graph, k_cache_data, q_projected);
  const ValueRef v_cache =
      prepack_standard_like(graph, v_cache_data, q_projected);

  update_cache_impl(graph, {k_projected, k_cache, input_pos_symint, -1});
  update_cache_impl(graph, {v_projected, v_cache, input_pos_symint, -1});

  sdpa_impl(
      graph,
      {q_projected,
       k_cache,
       v_cache,
       input_pos_symint,
       attn_mask,
       dropout_p,
       is_causal,
       scale,
       out});
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(update_cache.default, update_cache_impl);
  VK_REGISTER_OP(llama.custom_sdpa.default, sdpa_impl);
  VK_REGISTER_OP(llama.flash_attention.default, flash_attention_impl);
}

} // namespace vkcompute
