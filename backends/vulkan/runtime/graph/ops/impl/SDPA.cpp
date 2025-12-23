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

bool is_single_token(ComputeGraph* graph, const ValueRef& q_projected) {
  return graph->size_at<uint32_t>(-3, q_projected) == 1;
}

//
// Resize functions
//

void resize_compute_attn_weights_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights = args.at(0).refs.at(0);
  const ValueRef q_projected = args.at(1).refs.at(0);
  const ValueRef input_pos_symint = resize_args.at(0);

  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  const int32_t input_pos_val = graph->read_symint(input_pos_symint);

  const uint32_t context_len = seq_len + input_pos_val;

  std::vector<int64_t> out_sizes = {
      1, // batch
      num_q_heads,
      utils::align_up_4(seq_len),
      utils::align_up_4(context_len)};

  graph->virtual_resize(attn_weights, out_sizes);
}

void resize_sdpa_attn_weights_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights_softmax = args.at(0).refs.at(0);
  const ValueRef attn_weights = args.at(1).refs.at(0);

  graph->virtual_resize(attn_weights_softmax, graph->sizes_of(attn_weights));
}

void resize_sdpa_compute_out_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef q_projected = resize_args.at(0);

  graph->virtual_resize(out, graph->sizes_of(q_projected));
}

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

//
// Shader dispatch pick functions
//

utils::uvec3 kv_cache_update_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef projected = args.at(1).refs.at(0);

  const uint32_t head_dim_size = graph->size_at<uint32_t>(-1, projected);
  const uint32_t num_heads = graph->size_at<uint32_t>(-2, projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, projected);

  return {utils::div_up_4(head_dim_size), seq_len, num_heads};
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

vkapi::ShaderInfo pick_sdpa_compute_attn_weights_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef q_projected = args.at(1).refs.at(0);
  const ValueRef k_cache = args.at(1).refs.at(1);

  const bool is_gemv = is_single_token(graph, q_projected);

  std::string shader_name = "sdpa_compute_attn_weights";
  if (is_gemv) {
    shader_name += "_coop";
  } else {
    shader_name += "_tiled";
  }

  add_storage_type_suffix(shader_name, graph->storage_type_of(q_projected));
  add_storage_type_suffix(shader_name, graph->storage_type_of(k_cache));
  add_dtype_suffix(shader_name, graph->dtype_of(q_projected));

  return VK_KERNEL_FROM_STR(shader_name);
}

utils::uvec3 pick_sdpa_compute_attn_weights_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef q_projected = args.at(1).refs.at(0);
  const ValueRef input_pos_symint = resize_args.at(0);

  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  const int32_t input_pos_val = graph->read_symint(input_pos_symint);

  const uint32_t context_len = seq_len + input_pos_val;

  const uint32_t N4 = utils::div_up_4(context_len);
  const uint32_t M4 = utils::div_up_4(seq_len);

  return {N4, M4, num_q_heads};
}

utils::uvec3 pick_sdpa_compute_attn_weights_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {1, 64, 1};
  } else {
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
}

utils::uvec3 pick_sdpa_attn_weights_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef q_projected = resize_args.at(0);

  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  return {1, seq_len, num_q_heads};
}

utils::uvec3 pick_sdpa_attn_weights_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return {64, 1, 1};
}

vkapi::ShaderInfo pick_sdpa_compute_out_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef v_cache = args.at(1).refs.at(1);

  const ValueRef q_projected = resize_args.at(0);

  const bool is_gemv = is_single_token(graph, q_projected);

  std::string shader_name = "sdpa_compute_out";
  if (is_gemv) {
    shader_name += "_coop";
  } else {
    shader_name += "_tiled";
  }

  add_storage_type_suffix(shader_name, graph->storage_type_of(out));
  add_storage_type_suffix(shader_name, graph->storage_type_of(v_cache));
  add_dtype_suffix(shader_name, graph->dtype_of(out));

  return VK_KERNEL_FROM_STR(shader_name);
}

utils::uvec3 pick_sdpa_compute_out_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef q_projected = resize_args.at(0);

  const uint32_t head_dim = graph->size_at<uint32_t>(-1, q_projected);
  const uint32_t num_q_heads = graph->size_at<uint32_t>(-2, q_projected);
  const uint32_t seq_len = graph->size_at<uint32_t>(-3, q_projected);

  const uint32_t N4 = utils::div_up_4(head_dim);
  const uint32_t M4 = utils::div_up_4(seq_len);

  return {N4, M4, num_q_heads};
}

utils::uvec3 pick_sdpa_compute_out_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {1, 64, 1};
  } else {
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
}

//
// Dispatch nodes
//

void add_sdpa_kv_cache_update_node(
    ComputeGraph& graph,
    const ValueRef input_pos_symint,
    const ValueRef projected,
    const ValueRef cache) {
  std::string kernel_name("sdpa_kv_cache_update");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(cache));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(projected));
  add_dtype_suffix(kernel_name, graph.dtype_of(projected));

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(cache),
      graph.sizes_ubo(projected),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

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
      {input_pos_symint},
      // Resizing Logic
      nullptr));
}

void add_sdpa_compute_attn_weights_node(
    ComputeGraph& graph,
    const ValueRef q_projected,
    const ValueRef k_cache,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights) {
  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected),
      graph.sizes_ubo(k_cache),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_compute_attn_weights_shader,
      pick_sdpa_compute_attn_weights_global_wg_size,
      pick_sdpa_compute_attn_weights_local_wg_size,
      // Inputs and Outputs
      {{attn_weights, vkapi::kWrite}, {{q_projected, k_cache}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {scale_val},
      // Resize Args
      {input_pos_symint},
      // Resizing Logic
      resize_compute_attn_weights_node));
}

void add_sdpa_attn_weights_softmax_node(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q_projected,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax) {
  std::string shader_name = "sdpa_attn_weights_softmax";
  add_storage_type_suffix(
      shader_name, graph.storage_type_of(attn_weights_softmax));
  add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      pick_sdpa_attn_weights_softmax_global_wg_size,
      pick_sdpa_attn_weights_softmax_local_wg_size,
      // Inputs and Outputs
      {{attn_weights_softmax, vkapi::kWrite}, {attn_weights, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {q_projected, input_pos_symint},
      // Resizing Logic
      resize_sdpa_attn_weights_softmax_node));
}

void add_sdpa_compute_out_node(
    ComputeGraph& graph,
    const ValueRef attn_weights_softmax,
    const ValueRef v_cache,
    const ValueRef q_projected,
    const ValueRef input_pos_symint,
    const ValueRef out) {
  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q_projected),
      graph.sizes_ubo(v_cache),
      graph.get_or_create_int_param_buffer(input_pos_symint)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_compute_out_shader,
      pick_sdpa_compute_out_global_wg_size,
      pick_sdpa_compute_out_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{attn_weights_softmax, v_cache}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {q_projected, input_pos_symint},
      // Resizing Logic
      resize_sdpa_compute_out_node));
}

//
// High level operator impl
//

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

  add_sdpa_kv_cache_update_node(graph, input_pos_symint, value, cache);
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

  const int64_t num_q_heads = graph.size_at<int64_t>(-2, q_projected);
  int64_t max_seq_len = graph.size_at<int64_t>(-3, q_projected);
  const int64_t max_context_len = graph.size_at<int32_t>(-3, k_cache);

  const utils::StorageType attn_weights_storage =
      graph.storage_type_of(q_projected);

  // If using buffer storage for attn weights, we need to ensure that the buffer
  // numel limit is not exceeded. If needed, manually adjust max_seq_len based
  // on the buffer numel limit.
  if (attn_weights_storage == utils::kBuffer) {
    const int64_t max_buffer_numel = graph.max_buffer_numel();
    if (num_q_heads * max_seq_len * max_context_len >= max_buffer_numel) {
      // Compute the maximum possible value for max_seq_len that will hit
      // the buffer numel limit.
      max_seq_len = max_buffer_numel / (num_q_heads * max_context_len);
      // Adjust down to the nearest multiple of 4 to make sure the limit is
      // not hit.
      if (max_seq_len % 4 != 0) {
        max_seq_len = (max_seq_len / 4) * 4;
      } else {
        max_seq_len -= 4;
      }
    }
  }

  std::vector<int64_t> attn_weight_full_sizes = {
      1, // batch
      num_q_heads,
      max_seq_len,
      max_context_len};

  TmpTensor attn_weights(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  TmpTensor attn_weights_softmax(
      &graph,
      attn_weight_full_sizes,
      graph.dtype_of(q_projected),
      attn_weights_storage,
      utils::kWidthPacked);

  add_sdpa_compute_attn_weights_node(
      graph, q_projected, k_cache, input_pos_symint, attn_weights);

  add_sdpa_attn_weights_softmax_node(
      graph, attn_weights, q_projected, input_pos_symint, attn_weights_softmax);

  add_sdpa_compute_out_node(
      graph, attn_weights_softmax, v_cache, q_projected, input_pos_symint, out);
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

  utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      graph.add_tensor_like(k_cache_data, cache_storage, utils::kWidthPacked);
  const ValueRef v_cache =
      graph.add_tensor_like(v_cache_data, cache_storage, utils::kWidthPacked);

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

void compute_attn_weight_with_kv_cache_impl(
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
  (void)attn_mask;
  const ValueRef dropout_p = args[arg_idx++];
  (void)dropout_p;
  const ValueRef is_causal = args[arg_idx++];
  (void)is_causal;
  const ValueRef scale = args[arg_idx++];
  (void)scale;

  // Output tensors
  const ValueRef out = args[arg_idx++];

  (void)sequence_len;

  const utils::StorageType cache_storage = graph.storage_type_of(q_projected);
  const ValueRef k_cache =
      graph.add_tensor_like(k_cache_data, cache_storage, utils::kWidthPacked);
  const ValueRef v_cache =
      graph.add_tensor_like(v_cache_data, cache_storage, utils::kWidthPacked);

  update_cache_impl(graph, {k_projected, k_cache, input_pos_symint, -1});
  update_cache_impl(graph, {v_projected, v_cache, input_pos_symint, -1});

  add_sdpa_compute_attn_weights_node(
      graph, q_projected, k_cache, input_pos_symint, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(update_cache.default, update_cache_impl);
  VK_REGISTER_OP(llama.custom_sdpa.default, sdpa_impl);
  VK_REGISTER_OP(
      testing.compute_attn_weight_with_kv_cache.default,
      compute_attn_weight_with_kv_cache_impl);
}

} // namespace vkcompute
