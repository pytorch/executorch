/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_rotary_embedding_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);

  std::vector<int64_t> in_sizes = in->sizes();
  // UNCOMMENT BELOW IF NEEDED
  // out->virtual_resize(in_sizes);
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

  utils::uvec3 global_wg_size = graph.logical_limits_of(xq_out);
  global_wg_size[0] /= 2;
  const utils::uvec3 local_wg_size = graph.create_local_wg_size(global_wg_size);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      // Shader
      VK_KERNEL_FROM_STR(kernel_name),
      // Workgroup sizes
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{{xq_out, xk_out}, vkapi::kWrite},
       {{xq, xk, freqs_cos, freqs_sin}, vkapi::kRead}},
      // Parameter buffers
      {graph.logical_limits_ubo(xq_out), graph.logical_limits_ubo(xk_out)},
      // Specialization Constants
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
