/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q4gswLinear.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Resize d_x to d_out.shape[:-1] + (K,), mirroring linear_q4gsw_backward_meta.
// extra_args = { weight_data, d_out }.
void resize_linear_q4gsw_backward_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef d_x = args.at(0).refs.at(0);
  const ValueRef weight_data = extra_args.at(0);
  const ValueRef d_out = extra_args.at(1);
  const int64_t K = graph->sizes_of(weight_data).at(1) * 2;
  std::vector<int64_t> new_sizes = graph->sizes_of(d_out);
  new_sizes.back() = K;
  graph->virtual_resize(d_x, new_sizes);
}

utils::uvec3 linear_q4gsw_backward_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef d_x = args.at(0).refs.at(0);
  const uint32_t K = graph->size_at<uint32_t>(-1, d_x);
  const uint32_t M = utils::safe_downcast<uint32_t>(graph->numel_of(d_x) / K);
  const uint32_t tiles = utils::div_up_4(M) * utils::div_up_4(K);
  return {tiles, 1u, 1u};
}

utils::uvec3 linear_q4gsw_backward_local_wg_size(
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

void linear_q4gsw_backward(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t i = 0;
  const ValueRef d_out = args.at(i++);
  const ValueRef weight_data = args.at(i++);
  const ValueRef weight_scales_data = args.at(i++);
  const ValueRef group_size_ref = args.at(i++);
  const ValueRef d_x = args.at(i++);

  VK_CHECK_COND(graph.dtype_of(d_out) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(d_x) == vkapi::kFloat);
  VK_CHECK_COND(graph.is_buffer_storage(d_out));
  VK_CHECK_COND(graph.is_buffer_storage(d_x));

  const vkapi::ScalarType in_dtype = graph.dtype_of(d_out);
  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size_ref);
  VK_CHECK_COND(group_size_val > 0 && group_size_val % 4 == 0);

  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t N = weight_sizes.at(0);
  const int64_t K = weight_sizes.at(1) * 2;
  VK_CHECK_COND(N > 0 && K > 0);
  VK_CHECK_COND(N % 4 == 0 && K % 4 == 0);
  VK_CHECK_COND(K % group_size_val == 0);
  VK_CHECK_COND(graph.size_at<int64_t>(-1, d_out) == N);

  const ValueRef packed_weight = prepack_q4_w_4x8_nc_buffer(graph, weight_data);
  const ValueRef packed_scales =
      prepack_q4_scales(graph, weight_scales_data, in_dtype);

  const uint32_t M = utils::safe_downcast<uint32_t>(graph.numel_of(d_out) / N);
  const uint32_t tiles =
      utils::div_up_4(M) * utils::div_up_4(static_cast<uint32_t>(K));
  VK_CHECK_COND(
      (tiles + 63u) / 64u <= 65535u,
      "linear_q4gsw_backward: tile count exceeds max workgroup count");

  std::string kernel_name = "q4gsw_backward";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(d_x));
  add_dtype_suffix(kernel_name, graph.dtype_of(d_x));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      linear_q4gsw_backward_global_wg_size,
      linear_q4gsw_backward_local_wg_size,
      // Inputs and Outputs
      {{d_x, vkapi::kWrite},
       {{d_out, packed_weight, packed_scales}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(d_out), graph.sizes_ubo(d_x)},
      // Push Constants
      {},
      // Specialization Constants
      {static_cast<uint32_t>(group_size_val)},
      // Resize Args
      {weight_data, d_out},
      // Resizing Logic
      resize_linear_q4gsw_backward_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q4gsw_backward.default, linear_q4gsw_backward);
}

} // namespace vkcompute
