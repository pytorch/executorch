/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Permute.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

void add_t_default_node(ComputeGraph& graph, ValueRef in, ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);

  VK_CHECK_COND(check_memory_layout_is(*t_in, api::kChannelsPacked));

  // TODO: Verify 0-dim tensor
  VK_CHECK_COND(
      (1 <= t_in->dim()) && (t_in->dim() <= 2),
      "aten.t tensor must be 1d or 2d");

  std::vector<int64_t> permute_dims;
  if (t_in->dim() == 1) {
    permute_dims.emplace_back(0);
  } else {
    permute_dims.emplace_back(1);
    permute_dims.emplace_back(0);
  }

  add_permute_node(graph, in, permute_dims, out);
}

void t_default(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_t_default_node(graph, args[0], args[1]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.t.default, t_default);
}

} // namespace vkcompute
