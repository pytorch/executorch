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

void resize_linear_q4gsw_dw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef dW = args.at(0).refs.at(0);
  const ValueRef d_out = args.at(1).refs.at(0);
  const ValueRef x = args.at(1).refs.at(1);
  const int64_t N = graph->size_at<int64_t>(-1, d_out);
  const int64_t K = graph->size_at<int64_t>(-1, x);
  graph->virtual_resize(dW, {N, K});
}

utils::uvec3 linear_q4gsw_dw_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef dW = args.at(0).refs.at(0);
  const uint32_t N = graph->size_at<uint32_t>(-2, dW);
  const uint32_t K = graph->size_at<uint32_t>(-1, dW);
  const uint32_t tiles = utils::div_up_4(N) * utils::div_up_4(K);
  return {tiles, 1u, 1u};
}

utils::uvec3 linear_q4gsw_dw_local_wg_size(
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
  return {64u, 1u, 1u};
}

void linear_q4gsw_dw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t i = 0;
  const ValueRef d_out = args.at(i++);
  const ValueRef x = args.at(i++);
  const ValueRef dW = args.at(i++);

  VK_CHECK_COND(graph.dtype_of(d_out) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(x) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(dW) == vkapi::kFloat);
  VK_CHECK_COND(graph.is_buffer_storage(d_out));
  VK_CHECK_COND(graph.is_buffer_storage(x));
  VK_CHECK_COND(graph.is_buffer_storage(dW));

  const int64_t N = graph.size_at<int64_t>(-1, d_out);
  const int64_t K = graph.size_at<int64_t>(-1, x);
  VK_CHECK_COND(N > 0 && K > 0);
  VK_CHECK_COND(graph.numel_of(d_out) % N == 0);
  VK_CHECK_COND(graph.numel_of(x) % K == 0);
  VK_CHECK_COND(graph.numel_of(d_out) / N == graph.numel_of(x) / K);

  const uint32_t tiles = utils::div_up_4(static_cast<uint32_t>(N)) *
      utils::div_up_4(static_cast<uint32_t>(K));
  VK_CHECK_COND(
      (tiles + 63u) / 64u <= 65535u,
      "linear_q4gsw_dw: tile count exceeds max workgroup count");

  std::string kernel_name = "linear_q4gsw_dw";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(dW));
  add_dtype_suffix(kernel_name, graph.dtype_of(dW));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      linear_q4gsw_dw_global_wg_size,
      linear_q4gsw_dw_local_wg_size,
      // Inputs and Outputs
      {{dW, vkapi::kWrite}, {{d_out, x}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(d_out), graph.sizes_ubo(x)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_linear_q4gsw_dw_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q4gsw_dw.default, linear_q4gsw_dw);
}

} // namespace vkcompute
