/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_embedding_args(
    const api::vTensor& weight,
    const api::vTensor& in,
    const api::vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(weight, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(in, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, utils::kChannelsPacked));
}

void add_embedding_node(
    ComputeGraph& graph,
    ValueRef weight,
    ValueRef in,
    ValueRef out) {
  vTensorPtr t_weight = graph.get_tensor(weight);
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  check_embedding_args(*t_weight, *t_in, *t_out);

  std::string kernel_name = "embedding";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      {{out, vkapi::MemoryAccessType::WRITE},
       {{in, weight}, vkapi::MemoryAccessType::READ}},
      {t_out->sizes_ubo()}));
}

void embedding(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef weight = prepack_if_tensor_ref(graph, args[0]);
  ValueRef in = prepack_if_tensor_ref(graph, args[1]);
  ValueRef out = args[5];

  add_embedding_node(graph, weight, in, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.embedding.default, embedding);
}

} // namespace vkcompute
