/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Preprocess.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Global WG for transpose cast contig to vectorized.
// 1x1 (buffer):  {K, ceil(M/4), 1} — one vec4 per thread
// 4x4 (texture): {K/4, ceil(M/4), 1} — 4 vec4 per thread (full texel use)
//
// M and K are read from fp_input's live sizes (resize_args[0]) so that
// virtual_resize updates flow through. When M == 1 the transpose is a no-op
// (the downstream GEMV path reads fp_input directly) and global_wg returns
// {0,0,0} to make DispatchNode::encode() skip the recording entirely.
static utils::uvec3 transpose_cast_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;

  const ValueRef fp_input_ref = resize_args.at(0);
  std::vector<int64_t> in_sizes = graph->sizes_of(fp_input_ref);
  const uint32_t M = static_cast<uint32_t>(utils::val_at(-2, in_sizes));
  const uint32_t K = static_cast<uint32_t>(utils::val_at(-1, in_sizes));

  if (M == 1u) {
    return {0u, 0u, 0u};
  }

  bool is_4x4 = shader.kernel_name.find("4x4") != std::string::npos;
  if (is_4x4) {
    return {utils::div_up(K, 4u), utils::div_up(M, 4u), 1u};
  }
  return {K, utils::div_up(M, 4u), 1u};
}

static utils::uvec3 transpose_cast_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;

  bool is_4x4 = shader.kernel_name.find("4x4") != std::string::npos;
  return is_4x4 ? utils::uvec3{2u, 16u, 1u} : utils::uvec3{8u, 8u, 1u};
}

// Resize the transposed output tensor to match current fp_input dimensions.
// Shape is {K * ceil(M/4) * 4} — a flat vec4 buffer with M rounded up to 4.
static void resize_transpose_cast_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef fp_input_ref = resize_args.at(0);
  const ValueRef transposed_out = args.at(0).refs.at(0);
  std::vector<int64_t> in_sizes = graph->sizes_of(fp_input_ref);
  const int64_t M = utils::val_at(-2, in_sizes);
  const int64_t K = utils::val_at(-1, in_sizes);
  const int64_t M4 = (M + 3) / 4;

  graph->virtual_resize(transposed_out, {K * M4 * 4});
}

void add_transpose_cast_contig_to_vectorized_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef output) {
  bool is_texture_input = !graph.is_buffer_storage(fp_input);

  // Name pattern:
  // transpose_cast_contig_to_vectorized[_4x4]_{in_dtype}_{in_storage}_{out_dtype}_{out_storage}
  std::string kernel_name = "transpose_cast_contig_to_vectorized";
  if (is_texture_input) {
    kernel_name += "_4x4";
  }

  kernel_name +=
      (graph.dtype_of(fp_input) == vkapi::kHalf) ? "_half" : "_float";
  kernel_name += is_texture_input ? "_texture3d" : "_buffer";
  kernel_name += (graph.dtype_of(output) == vkapi::kHalf) ? "_half" : "_float";
  kernel_name += graph.is_buffer_storage(output) ? "_buffer" : "_texture2d";

  // Bind the input sizes UBO directly from fp_input so the shader reads M/K
  // from the tensor's live metadata (which is updated by virtual_resize()).
  // For 2D [M, K] input, `sizes_ubo` emits {K, M, 1, 1} in WHCN order, which
  // is exactly what the shader's `sizes.x`, `sizes.y` expect.
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      transpose_cast_global_wg_size,
      transpose_cast_local_wg_size,
      {{output, vkapi::kWrite}, {fp_input, vkapi::kRead}},
      {graph.sizes_ubo(fp_input)},
      {},
      {},
      // resize_args[0] = fp_input: drives both self-gating (M==1 → {0,0,0})
      // and resize_transpose_cast_node (virtual_resize of transposed output).
      {fp_input},
      resize_transpose_cast_node));
}

} // namespace vkcompute
