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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <cmath>

namespace vkcompute {

namespace {

//
// SDPA mode: distinguishes the two dispatch families sharing this file.
//   LLM   — Llama-style KV-cache SDPA. Q layout [B=1, S, H,    D] (DHSB).
//           Separate k_cache/v_cache inputs + input_pos_symint for dynamic
//           context_len. attn_weights are padded to multiples of 4 in the
//           S/context_len dims and carry the input dtype. A coop (GEMV)
//           shader variant is selected for single-token decode.
//   FUSED — General SDPA fused op. Q layout [B, H, S, D] (DSHB). No cache,
//           optional additive attn_mask, optional scale arg. attn_weights
//           are unpadded and always fp32. Tiled shader variant only.
//
enum class SDPAMode { LLM, FUSED };

//
// Common dimension helper: folds the axis-swap for LLM vs fused Q layouts.
// `input_pos_symint` is used only for LLM (context_len = S + input_pos);
// pass kDummyValueRef for FUSED.
//
struct SDPADims {
  int64_t B = 1;
  int64_t H = 0;
  int64_t S = 0;
  int64_t D = 0;
  int64_t context_len = 0; // LLM: S + input_pos_val; FUSED: size_at(-2, k)
  int64_t max_context_len = 0; // LLM: size_at(-3, k); FUSED: size_at(-2, k)
};

} // namespace

SDPADims compute_sdpa_dims(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const SDPAMode mode) {
  SDPADims d;
  d.D = graph.size_at<int64_t>(-1, q);
  if (mode == SDPAMode::LLM) {
    // Q: [B=1, S, H, D] (DHSB), K: [B=1, C_max, H_kv, D]
    // `k` may be kDummyValueRef in dispatch pickers that don't need it;
    // max_context_len is only read when k is valid.
    d.B = 1;
    d.H = graph.size_at<int64_t>(-2, q);
    d.S = graph.size_at<int64_t>(-3, q);
    d.max_context_len = is_valid(k) ? graph.size_at<int64_t>(-3, k) : 0;
    const int32_t input_pos_val =
        is_valid(input_pos_symint) ? graph.read_symint(input_pos_symint) : 0;
    d.context_len = d.S + input_pos_val;
  } else {
    // Q: [B, H, S, D] (DSHB), K: [B, H_kv, L, D]
    d.B = graph.size_at<int64_t>(-4, q);
    d.H = graph.size_at<int64_t>(-3, q);
    d.S = graph.size_at<int64_t>(-2, q);
    d.context_len = graph.size_at<int64_t>(-2, k);
    d.max_context_len = d.context_len;
  }
  return d;
}

bool is_single_token(ComputeGraph* graph, const ValueRef& q_projected) {
  return graph->size_at<uint32_t>(-3, q_projected) == 1;
}

//
// Resize functions
//

// Unified attn_weights resize. In LLM mode the shape is padded to multiples of
// 4 in the S/context_len dims (to match the tiled shader's iteration space);
// in fused mode it's the unpadded [B, H, S, L].
// resize_args layout: [q, k, input_pos_symint_or_dummy, mode_as_int]
void resize_sdpa_attn_weights_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights = args.at(0).refs.at(0);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPAMode mode = static_cast<SDPAMode>(resize_args.at(3));

  std::vector<int64_t> out_sizes;
  if (mode == SDPAMode::LLM) {
    const int64_t num_q_heads = graph->size_at<int64_t>(-2, q);
    const int64_t seq_len = graph->size_at<int64_t>(-3, q);
    const int32_t input_pos_val = graph->read_symint(input_pos_symint);
    const int64_t context_len = seq_len + input_pos_val;
    out_sizes = {
        1,
        num_q_heads,
        static_cast<int64_t>(utils::align_up_4(seq_len)),
        static_cast<int64_t>(utils::align_up_4(context_len))};
  } else {
    const int64_t B = graph->size_at<int64_t>(-4, q);
    const int64_t H = graph->size_at<int64_t>(-3, q);
    const int64_t S = graph->size_at<int64_t>(-2, q);
    const int64_t L = graph->size_at<int64_t>(-2, k);
    out_sizes = {B, H, S, L};
  }
  graph->virtual_resize(attn_weights, out_sizes);
}

// Softmax preserves attn_weights shape exactly; identical across modes.
void resize_sdpa_attn_weights_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef attn_weights_softmax = args.at(0).refs.at(0);
  const ValueRef attn_weights = args.at(1).refs.at(0);

  graph->virtual_resize(attn_weights_softmax, graph->sizes_of(attn_weights));
}

// Out matches Q's shape in both modes. resize_args[0] = q.
void resize_sdpa_out_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef q = resize_args.at(0);

  graph->virtual_resize(out, graph->sizes_of(q));
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

// resize_args layout for SDPA dispatch pickers mirrors the node creation
// helper: [q, k, input_pos_symint_or_dummy, mode_as_int].
static inline SDPAMode mode_of(const std::vector<ValueRef>& resize_args) {
  return static_cast<SDPAMode>(resize_args.at(3));
}

vkapi::ShaderInfo pick_sdpa_qk_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const ValueRef q_projected = args.at(1).refs.at(0);
    const ValueRef k_cache = args.at(1).refs.at(1);
    const bool is_gemv = is_single_token(graph, q_projected);

    std::string shader_name = "sdpa_compute_attn_weights";
    shader_name += is_gemv ? "_coop" : "_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(q_projected));
    add_storage_type_suffix(shader_name, graph->storage_type_of(k_cache));
    add_dtype_suffix(shader_name, graph->dtype_of(q_projected));
    return VK_KERNEL_FROM_STR(shader_name);
  } else {
    const ValueRef q = args.at(1).refs.at(0);
    const ValueRef k = args.at(1).refs.at(1);
    // Fused path uses bias variant iff attn_mask was provided (signalled via
    // 3 inputs in the read group: q, k, attn_mask).
    const bool has_bias = args.at(1).refs.size() >= 3;
    std::string shader_name =
        has_bias ? "fused_sdpa_qk_tiled_bias" : "fused_sdpa_qk_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(q));
    add_storage_type_suffix(shader_name, graph->storage_type_of(k));
    add_dtype_suffix(shader_name, graph->dtype_of(q));
    return VK_KERNEL_FROM_STR(shader_name);
  }
}

utils::uvec3 pick_sdpa_qk_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPADims d = compute_sdpa_dims(*graph, q, k, input_pos_symint, mode);

  // Dispatch grid: (context_len tiles, S tiles, H * B).
  const uint32_t N4 = utils::div_up_4(static_cast<uint32_t>(d.context_len));
  const uint32_t M4 = utils::div_up_4(static_cast<uint32_t>(d.S));
  return {N4, M4, static_cast<uint32_t>(d.H * d.B)};
}

utils::uvec3 pick_sdpa_qk_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const bool use_coop_algorithm =
        shader.kernel_name.find("_coop") != std::string::npos;
    if (use_coop_algorithm) {
      return {1, 64, 1};
    }
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
  return default_pick_local_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

utils::uvec3 pick_sdpa_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  // LLM reads H from axis -2, fused from axis -3 (handled by
  // compute_sdpa_dims).
  const int64_t num_q_heads = (mode == SDPAMode::LLM)
      ? graph->size_at<int64_t>(-2, q)
      : graph->size_at<int64_t>(-3, q);
  const int64_t seq_len = (mode == SDPAMode::LLM)
      ? graph->size_at<int64_t>(-3, q)
      : graph->size_at<int64_t>(-2, q);
  const int64_t B =
      (mode == SDPAMode::LLM) ? 1 : graph->size_at<int64_t>(-4, q);
  return {
      1,
      static_cast<uint32_t>(seq_len),
      static_cast<uint32_t>(num_q_heads * B)};
}

utils::uvec3 pick_sdpa_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;
  return {64, 1, 1};
}

vkapi::ShaderInfo pick_sdpa_av_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const ValueRef out = args.at(0).refs.at(0);
    const ValueRef v_cache = args.at(1).refs.at(1);
    const ValueRef q_projected = resize_args.at(0);
    const bool is_gemv = is_single_token(graph, q_projected);

    std::string shader_name = "sdpa_compute_out";
    shader_name += is_gemv ? "_coop" : "_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(out));
    add_storage_type_suffix(shader_name, graph->storage_type_of(v_cache));
    add_dtype_suffix(shader_name, graph->dtype_of(out));
    return VK_KERNEL_FROM_STR(shader_name);
  } else {
    const ValueRef out = args.at(0).refs.at(0);
    const ValueRef v = args.at(1).refs.at(1);
    std::string shader_name = "fused_sdpa_av_tiled";
    add_storage_type_suffix(shader_name, graph->storage_type_of(out));
    add_storage_type_suffix(shader_name, graph->storage_type_of(v));
    add_dtype_suffix(shader_name, graph->dtype_of(out));
    return VK_KERNEL_FROM_STR(shader_name);
  }
}

utils::uvec3 pick_sdpa_av_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const SDPAMode mode = mode_of(resize_args);
  const ValueRef q = resize_args.at(0);
  const ValueRef k = resize_args.at(1);
  const ValueRef input_pos_symint = resize_args.at(2);
  const SDPADims d = compute_sdpa_dims(*graph, q, k, input_pos_symint, mode);

  const uint32_t N4 = utils::div_up_4(static_cast<uint32_t>(d.D));
  const uint32_t M4 = utils::div_up_4(static_cast<uint32_t>(d.S));
  return {N4, M4, static_cast<uint32_t>(d.H * d.B)};
}

utils::uvec3 pick_sdpa_av_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const SDPAMode mode = mode_of(resize_args);
  if (mode == SDPAMode::LLM) {
    const bool use_coop_algorithm =
        shader.kernel_name.find("_coop") != std::string::npos;
    if (use_coop_algorithm) {
      return {1, 64, 1};
    }
    return pick_hw_square_wg_size(
        graph, shader, global_workgroup_size, args, resize_args);
  }
  return default_pick_local_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
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

// Unified QK node (attn_weights = scale * Q @ K^T [+ bias]).
// LLM: pass input_pos_symint (real symint), attn_mask = kDummyValueRef.
// FUSED: pass input_pos_symint = kDummyValueRef, attn_mask = valid ref or
//        kDummyValueRef to indicate no bias. scale_val is always passed as
//        a spec const; the LLM path computes it per head_dim and FUSED may
//        inherit from the caller-supplied scale.
void add_sdpa_compute_attn_weights_node(
    ComputeGraph& graph,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_mask,
    const float scale_val,
    const ValueRef attn_weights,
    const SDPAMode mode) {
  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(q),
      graph.sizes_ubo(k),
  };
  std::vector<ValueRef> read_inputs = {q, k};

  if (mode == SDPAMode::LLM) {
    param_ubos.append(graph.get_or_create_int_param_buffer(input_pos_symint));
  } else if (is_valid(attn_mask)) {
    param_ubos.append(graph.sizes_ubo(attn_mask));
    read_inputs.push_back(attn_mask);
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_qk_shader,
      pick_sdpa_qk_global_wg_size,
      pick_sdpa_qk_local_wg_size,
      // Inputs and Outputs
      {{attn_weights, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {scale_val},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode]
      {q, k, input_pos_symint, mode_ref},
      // Resizing Logic
      resize_sdpa_attn_weights_node));
}

void add_sdpa_attn_weights_softmax_node(
    ComputeGraph& graph,
    const ValueRef attn_weights,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef attn_weights_softmax,
    const SDPAMode mode) {
  std::string shader_name;
  if (mode == SDPAMode::LLM) {
    shader_name = "sdpa_attn_weights_softmax";
    add_storage_type_suffix(
        shader_name, graph.storage_type_of(attn_weights_softmax));
    add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));
  } else {
    shader_name = "fused_sdpa_softmax";
    add_storage_type_suffix(
        shader_name, graph.storage_type_of(attn_weights_softmax));
    add_dtype_suffix(shader_name, graph.dtype_of(attn_weights_softmax));
  }

  vkapi::ParamsBindList param_ubos;
  if (mode == SDPAMode::LLM) {
    param_ubos = {
        graph.sizes_ubo(q),
        graph.sizes_ubo(k),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    param_ubos = {graph.sizes_ubo(q), graph.sizes_ubo(k)};
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      pick_sdpa_softmax_global_wg_size,
      pick_sdpa_softmax_local_wg_size,
      // Inputs and Outputs
      {{attn_weights_softmax, vkapi::kWrite}, {attn_weights, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode]
      {q, k, input_pos_symint, mode_ref},
      // Resizing Logic
      resize_sdpa_attn_weights_softmax_node));
}

void add_sdpa_compute_out_node(
    ComputeGraph& graph,
    const ValueRef attn_weights_softmax,
    const ValueRef v,
    const ValueRef q,
    const ValueRef k,
    const ValueRef input_pos_symint,
    const ValueRef out,
    const SDPAMode mode) {
  vkapi::ParamsBindList param_ubos;
  if (mode == SDPAMode::LLM) {
    param_ubos = {
        graph.sizes_ubo(q),
        graph.sizes_ubo(v),
        graph.get_or_create_int_param_buffer(input_pos_symint)};
  } else {
    param_ubos = {graph.sizes_ubo(q), graph.sizes_ubo(v)};
  }

  const ValueRef mode_ref = static_cast<ValueRef>(mode);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_sdpa_av_shader,
      pick_sdpa_av_global_wg_size,
      pick_sdpa_av_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{attn_weights_softmax, v}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args: [q, k, input_pos_symint_or_dummy, mode]
      {q, k, input_pos_symint, mode_ref},
      // Resizing Logic
      resize_sdpa_out_node));
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

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  add_sdpa_compute_attn_weights_node(
      graph,
      q_projected,
      k_cache,
      input_pos_symint,
      /*attn_mask=*/kDummyValueRef,
      scale_val,
      attn_weights,
      SDPAMode::LLM);

  add_sdpa_attn_weights_softmax_node(
      graph,
      attn_weights,
      q_projected,
      k_cache,
      input_pos_symint,
      attn_weights_softmax,
      SDPAMode::LLM);

  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v_cache,
      q_projected,
      /*k=*/kDummyValueRef,
      input_pos_symint,
      out,
      SDPAMode::LLM);
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
  const ValueRef out = args[arg_idx];

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

  const int32_t head_dim_size = graph.size_at<int32_t>(-1, q_projected);
  const float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim_size));

  add_sdpa_compute_attn_weights_node(
      graph,
      q_projected,
      k_cache,
      input_pos_symint,
      /*attn_mask=*/kDummyValueRef,
      scale_val,
      out,
      SDPAMode::LLM);
}

//
// Fused SDPA entry point (et_vk.sdpa.default).
//
// Accepts pre-reshaped [B, H, S, D] tensors (DSHB) plus optional additive
// attn_mask and optional scale scalar. No KV cache; this is the general SDPA
// fused op used by non-LLM models.
//
void fused_sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef q = args[arg_idx++];
  const ValueRef k = args[arg_idx++];
  const ValueRef v = args[arg_idx++];
  const ValueRef attn_mask = args[arg_idx++];
  const ValueRef scale_ref = args[arg_idx++];
  const ValueRef out = args[arg_idx];

  // Validate inputs
  VK_CHECK_COND(graph.dim_of(q) == 4);
  VK_CHECK_COND(graph.dim_of(k) == 4);
  VK_CHECK_COND(graph.dim_of(v) == 4);
  // Head dim must match between Q and K
  VK_CHECK_COND(graph.size_at<int32_t>(-1, q) == graph.size_at<int32_t>(-1, k));
  // K and V must have same sequence length
  VK_CHECK_COND(graph.size_at<int32_t>(-2, k) == graph.size_at<int32_t>(-2, v));
  // All tensors must be width-packed
  VK_CHECK_COND(graph.packed_dim_of(q) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(k) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(v) == WHCN::kWidthDim);

  // Compute scale
  const int32_t head_dim = graph.size_at<int32_t>(-1, q);
  float scale_val;
  if (graph.val_is_none(scale_ref)) {
    scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim));
  } else {
    scale_val = graph.extract_scalar<float>(scale_ref);
  }

  // Resolve attn_mask: a None value is normalized to kDummyValueRef so the
  // unified helpers can branch with a single `is_valid()` check.
  const ValueRef attn_mask_ref =
      graph.val_is_none(attn_mask) ? kDummyValueRef : attn_mask;

  // Get dimensions for intermediate allocation
  const int64_t B = graph.size_at<int64_t>(-4, q);
  const int64_t H = graph.size_at<int64_t>(-3, q);
  const int64_t S = graph.size_at<int64_t>(-2, q);
  const int64_t L = graph.size_at<int64_t>(-2, k);

  std::vector<int64_t> attn_weight_sizes = {B, H, S, L};

  // attn_weights and attn_weights_softmax follow the output's storage so the
  // entire fused SDPA pipeline uses a uniform storage type. attn_weights stays
  // in fp32 for numerical stability of the Q@K^T accumulation.
  const utils::StorageType attn_storage = graph.storage_type_of(out);

  TmpTensor attn_weights(
      &graph,
      attn_weight_sizes,
      vkapi::ScalarType::Float,
      attn_storage,
      utils::kWidthPacked);

  TmpTensor attn_weights_softmax(
      &graph,
      attn_weight_sizes,
      graph.dtype_of(q),
      attn_storage,
      utils::kWidthPacked);

  // Phase 1: Q @ K^T with fp32 accumulation, apply scale and optional bias
  add_sdpa_compute_attn_weights_node(
      graph,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      attn_mask_ref,
      scale_val,
      attn_weights,
      SDPAMode::FUSED);

  // Phase 2: Softmax in fp32, output in input dtype
  add_sdpa_attn_weights_softmax_node(
      graph,
      attn_weights,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      attn_weights_softmax,
      SDPAMode::FUSED);

  // Phase 3: attn_weights_softmax @ V
  add_sdpa_compute_out_node(
      graph,
      attn_weights_softmax,
      v,
      q,
      k,
      /*input_pos_symint=*/kDummyValueRef,
      out,
      SDPAMode::FUSED);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
  VK_REGISTER_OP(update_cache.default, update_cache_impl);
  VK_REGISTER_OP(llama.custom_sdpa.default, sdpa_impl);
  VK_REGISTER_OP(
      testing.compute_attn_weight_with_kv_cache.default,
      compute_attn_weight_with_kv_cache_impl);
  VK_REGISTER_OP(et_vk.sdpa.default, fused_sdpa_impl);
}

} // namespace vkcompute
