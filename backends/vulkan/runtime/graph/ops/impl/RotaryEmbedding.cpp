/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_rotary_embedding_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef xq_out = args.at(0).refs.at(0);
  const ValueRef xk_out = args.at(0).refs.at(1);

  const ValueRef xq = args.at(1).refs.at(0);
  const ValueRef xk = args.at(1).refs.at(1);

  const std::vector<int64_t> xq_sizes = graph->sizes_of(xq);
  const std::vector<int64_t> xk_sizes = graph->sizes_of(xk);

  graph->virtual_resize(xq_out, xq_sizes);
  graph->virtual_resize(xk_out, xk_sizes);
}

utils::uvec3 rotary_embedding_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef xq_out = args.at(0).refs.at(0);

  // Head dim texel size
  const uint32_t D4 = utils::div_up_4(graph->size_at<uint32_t>(-1, xq_out));
  // Divide by 2 since each invocation computes 2 output locations
  const uint32_t D8 = utils::div_up(D4, uint32_t(2));

  // Number of query heads
  const uint32_t QH = graph->size_at<uint32_t>(-2, xq_out);
  // Input tokens sequence length
  const uint32_t S = graph->size_at<uint32_t>(-3, xq_out);

  return {D8, QH, S};
}

void add_rotary_embedding_node(
    ComputeGraph& graph,
    const ValueRef xq,
    const ValueRef xk,
    const ValueRef freqs_cos,
    const ValueRef freqs_sin,
    const ValueRef xq_out,
    const ValueRef xk_out) {
  VK_CHECK_COND(graph.size_at<int>(-1, xq) == graph.size_at<int>(-1, xk));
  VK_CHECK_COND(graph.size_at<int>(-3, xq) == graph.size_at<int>(-3, xk));
  VK_CHECK_COND(
      graph.size_at<int>(-1, xq) == graph.size_at<int>(-1, freqs_cos) * 2);
  VK_CHECK_COND(graph.sizes_of(freqs_cos) == graph.sizes_of(freqs_sin));

  VK_CHECK_COND(graph.packed_dim_of(xq) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(xk) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(freqs_cos) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(freqs_sin) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.has_standard_axis_map(xq));
  VK_CHECK_COND(graph.has_standard_axis_map(xk));
  VK_CHECK_COND(graph.has_standard_axis_map(freqs_cos));
  VK_CHECK_COND(graph.has_standard_axis_map(freqs_sin));

  std::string kernel_name = "rotary_embedding";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(xq_out));
  add_dtype_suffix(kernel_name, graph.dtype_of(xq_out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(xq_out),
      graph.meta_ubo(xk_out),
      graph.meta_ubo(freqs_cos)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      rotary_embedding_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{{xq_out, xk_out}, vkapi::kWrite},
       {{xq, xk, freqs_cos, freqs_sin}, vkapi::kRead}},
      // Parameter buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(xq_out), graph.hashed_layout_of(freqs_cos)},
      // Resize Args
      {},
      // Resizing Logic
      resize_rotary_embedding_node));
}

void apply_rotary_emb(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueListPtr out_tuple = graph.get_value_list(args[4]);
  const ValueRef xq_out = out_tuple->at(0);
  const ValueRef xk_out = out_tuple->at(1);

  add_rotary_embedding_node(
      graph, args[0], args[1], args[2], args[3], xq_out, xk_out);
}

//
// HuggingFace RoPE variant
//

utils::uvec3 rotary_embedding_hf_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef xq_out = args.at(0).refs.at(0);

  // Each invocation handles one texel (4 elements) along head_dim.
  // Dispatch for all head_dim elements so that both the rotary region and the
  // passthrough region (partial_rotary_factor < 1.0) are covered.
  const uint32_t D4 = utils::div_up_4(graph->size_at<uint32_t>(-1, xq_out));

  const uint32_t QH = graph->size_at<uint32_t>(-2, xq_out);
  const uint32_t S = graph->size_at<uint32_t>(-3, xq_out);

  return {D4, QH, S};
}

void add_rotary_embedding_hf_node(
    ComputeGraph& graph,
    const ValueRef xq,
    const ValueRef xk,
    const ValueRef freqs_cos,
    const ValueRef freqs_sin,
    const ValueRef start_pos,
    const ValueRef xq_out,
    const ValueRef xk_out) {
  VK_CHECK_COND(graph.size_at<int>(-1, xq) == graph.size_at<int>(-1, xk));
  VK_CHECK_COND(graph.size_at<int>(-3, xq) == graph.size_at<int>(-3, xk));
  // HF convention: freqs rotary_dim <= head_dim (supports
  // partial_rotary_factor)
  VK_CHECK_COND(
      graph.size_at<int>(-1, freqs_cos) <= graph.size_at<int>(-1, xq));
  // freqs_cos rotary_dim must be even (pairs required for rotation)
  VK_CHECK_COND(graph.size_at<int>(-1, freqs_cos) % 8 == 0);
  VK_CHECK_COND(graph.sizes_of(freqs_cos) == graph.sizes_of(freqs_sin));
  // freqs dim 0 is max_seq_len which must be >= current seq_len
  VK_CHECK_COND(
      graph.size_at<int>(-2, freqs_cos) >= graph.size_at<int>(-3, xq));

  VK_CHECK_COND(graph.packed_dim_of(xq) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(xk) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(freqs_cos) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(freqs_sin) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.has_standard_axis_map(xq));
  VK_CHECK_COND(graph.has_standard_axis_map(xk));
  VK_CHECK_COND(graph.has_standard_axis_map(freqs_cos));
  VK_CHECK_COND(graph.has_standard_axis_map(freqs_sin));

  const int32_t partial_rotary =
      graph.size_at<int>(-1, freqs_cos) < graph.size_at<int>(-1, xq) ? 1 : 0;

  std::string kernel_name = "rotary_embedding_hf";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(xq_out));
  add_dtype_suffix(kernel_name, graph.dtype_of(xq_out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(xq_out),
      graph.meta_ubo(xk_out),
      graph.meta_ubo(freqs_cos),
      graph.get_or_create_int_param_buffer(start_pos)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      rotary_embedding_hf_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{{xq_out, xk_out}, vkapi::kWrite},
       {{xq, xk, freqs_cos, freqs_sin}, vkapi::kRead}},
      // Parameter buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(xq_out),
       graph.hashed_layout_of(freqs_cos),
       partial_rotary},
      // Resize Args
      {},
      // Resizing Logic
      resize_rotary_embedding_node));
}

void apply_rotary_emb_hf(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueListPtr out_tuple = graph.get_value_list(args[5]);
  const ValueRef xq_out = out_tuple->at(0);
  const ValueRef xk_out = out_tuple->at(1);

  add_rotary_embedding_hf_node(
      graph, args[0], args[1], args[2], args[3], args[4], xq_out, xk_out);
}

//
// EdgeTAM-style RoPE variant with fused [cos, sin] freqs tensor
//
// Operates on a single tensor (Q or K) of shape [B, N, C] with pair-interleaved
// (real, imag) components along the last dim, and a freqs tensor with a total
// element count of N * C that packs (cos, sin) pairs in the same interleaved
// order as the x tensor. The freqs tensor may be passed in at any rank whose
// flattened layout is [N, C] — e.g. 2D `[N, C]` or 4D `[1, N, C/2, 2]`. This
// avoids callers having to emit a `view` dispatch (view_copy) purely to
// normalize rank.
//

void resize_rotary_embedding_interleaved_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  graph->virtual_resize(out, graph->sizes_of(in));
}

utils::uvec3 rotary_embedding_interleaved_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);

  const std::vector<int64_t> out_sizes = graph->sizes_of(out);
  VK_CHECK_COND(out_sizes.size() == 3);

  const uint32_t B = static_cast<uint32_t>(out_sizes.at(0));
  const uint32_t N = static_cast<uint32_t>(out_sizes.at(1));
  const uint32_t C = static_cast<uint32_t>(out_sizes.at(2));

  // One thread per output texel of 4 elements along C.
  return {utils::div_up_4(C), N, B};
}

void add_rotary_embedding_interleaved_node(
    ComputeGraph& graph,
    const ValueRef x,
    const ValueRef freqs_cis,
    const ValueRef out) {
  const std::vector<int64_t> x_sizes = graph.sizes_of(x);
  const std::vector<int64_t> freqs_sizes = graph.sizes_of(freqs_cis);

  VK_CHECK_COND(x_sizes.size() == 3);
  VK_CHECK_COND(x_sizes.at(2) % 4 == 0);

  // freqs_cis may arrive at any rank (commonly 2D [N, C] or 4D [1, N, C/2, 2]
  // from `torch.view_as_real(...).unsqueeze(0)`). Validate via numel rather
  // than per-dim equality so callers do not need to emit a view_copy purely
  // to flatten the shape.
  int64_t freqs_numel = 1;
  for (const int64_t s : freqs_sizes) {
    freqs_numel *= s;
  }
  const int64_t expected_numel = x_sizes.at(1) * x_sizes.at(2);
  VK_CHECK_COND(freqs_numel == expected_numel);

  VK_CHECK_COND(graph.packed_dim_of(x) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.has_standard_axis_map(x));
  VK_CHECK_COND(graph.has_standard_axis_map(out));
  // freqs_cis is pinned to buffer storage via op_registry so the shader can
  // use flat (row, col) indexing regardless of its declared rank.
  VK_CHECK_COND(graph.is_buffer_storage(freqs_cis));

  std::string kernel_name = "apply_rotary_emb_interleaved";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      rotary_embedding_interleaved_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{x, freqs_cis}, vkapi::kRead}},
      // Parameter buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out)},
      // Resize Args
      {},
      // Resizing Logic
      resize_rotary_embedding_interleaved_node));
}

void apply_rotary_emb_interleaved(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  add_rotary_embedding_interleaved_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.apply_rotary_emb.default, apply_rotary_emb);
  VK_REGISTER_OP(et_vk.apply_rotary_emb_hf.default, apply_rotary_emb_hf);
  VK_REGISTER_OP(
      et_vk.apply_rotary_emb_interleaved.default, apply_rotary_emb_interleaved);
}

} // namespace vkcompute
