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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Resize the packed output to the W_4X8 int buffer size K4 * N4_padded * 2,
// matching prepack_q4_w_4x8_nc_buffer's layout. extra_args = { latent }.
void resize_q4gsw_requant_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef packed = args.at(0).refs.at(0);
  const ValueRef latent = extra_args.at(0);
  const std::vector<int64_t> latent_sizes = graph->sizes_of(latent);
  const int64_t N = latent_sizes.at(0);
  const int64_t K = latent_sizes.at(1);
  const int64_t K4 = K / 4;
  const int64_t N4 = N / 4;
  const int64_t N4_padded = (N4 + 1) & ~int64_t{1};
  graph->virtual_resize(packed, {K4 * N4_padded * 2});
}

utils::uvec3 q4gsw_requant_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef latent = args.at(1).refs.at(0);
  const std::vector<int64_t> latent_sizes = graph->sizes_of(latent);
  const uint32_t N = utils::safe_downcast<uint32_t>(latent_sizes.at(0));
  const uint32_t K = utils::safe_downcast<uint32_t>(latent_sizes.at(1));
  const uint32_t K4 = K / 4u;
  const uint32_t N4 = (N + 3u) / 4u;
  const uint32_t N8 = (N4 + 1u) / 2u;
  return {K4, N8, 1u};
}

utils::uvec3 q4gsw_requant_local_wg_size(
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
  return {8u, 8u, 1u};
}

void q4gsw_requant(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t i = 0;
  const ValueRef latent = args.at(i++);
  const ValueRef scales = args.at(i++);
  const ValueRef group_size_ref = args.at(i++);
  const ValueRef packed = args.at(i++);

  VK_CHECK_COND(graph.dtype_of(latent) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(scales) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(packed) == vkapi::kInt);
  VK_CHECK_COND(graph.is_buffer_storage(latent));
  VK_CHECK_COND(graph.is_buffer_storage(packed));

  const int64_t group_size_val = graph.extract_scalar<int64_t>(group_size_ref);
  VK_CHECK_COND(group_size_val > 0 && group_size_val % 4 == 0);

  const std::vector<int64_t> latent_sizes = graph.sizes_of(latent);
  VK_CHECK_COND(latent_sizes.size() == 2);
  const int64_t N = latent_sizes.at(0);
  const int64_t K = latent_sizes.at(1);
  VK_CHECK_COND(N > 0 && K > 0);
  VK_CHECK_COND(N % 4 == 0 && K % 4 == 0);
  VK_CHECK_COND(K % group_size_val == 0);

  const uint32_t K4 = utils::safe_downcast<uint32_t>(K / 4);
  const uint32_t N4 = utils::safe_downcast<uint32_t>(N / 4);
  const uint32_t N8 = (N4 + 1u) / 2u;
  VK_CHECK_COND(
      K4 <= 65535u && N8 <= 65535u,
      "q4gsw_requant: dispatch grid exceeds max workgroup count");

  // Scales are a frozen constant; materialize them to a GPU buffer once.
  const ValueRef packed_scales =
      prepack_standard(graph, scales, utils::kBuffer, utils::kWidthPacked);

  std::string kernel_name = "q4gsw_requant";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(latent));
  add_dtype_suffix(kernel_name, graph.dtype_of(latent));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      q4gsw_requant_global_wg_size,
      q4gsw_requant_local_wg_size,
      // Inputs and Outputs
      {{packed, vkapi::kWrite}, {{latent, packed_scales}, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(latent)},
      // Push Constants
      {},
      // Specialization Constants
      {static_cast<uint32_t>(group_size_val)},
      // Resize Args
      {latent},
      // Resizing Logic
      resize_q4gsw_requant_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q4gsw_requant.default, q4gsw_requant);
}

} // namespace vkcompute
