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

  utils::uvec3 global_wg_size = graph->logical_limits_of(xq_out);
  global_wg_size[0] /= 2;

  return global_wg_size;
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
  add_dtype_suffix(kernel_name, graph.dtype_of(xq_out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      rotary_embedding_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{{xq_out, xk_out}, vkapi::kWrite},
       {{xq, xk, freqs_cos, freqs_sin}, vkapi::kRead}},
      // Parameter buffers
      {graph.logical_limits_ubo(xq_out), graph.logical_limits_ubo(xk_out)},
      // Push Constants
      {},
      // Specialization Constants
      {},
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

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.apply_rotary_emb.default, apply_rotary_emb);
}

} // namespace vkcompute
