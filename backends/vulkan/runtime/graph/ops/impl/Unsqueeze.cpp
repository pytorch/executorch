/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Permute.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_unsqueeze_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef dim_ref,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  VK_CHECK_COND(
      t_in->dim() < 4, "Cannot unsqueeze a tensor with more than 3 dimensions");

  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  int64_t out_dim = t_out->dim();

  std::vector<int64_t> permute_dims(out_dim);
  for (int i = 1; i <= dim; i++) {
    permute_dims[i - 1] = i;
  }
  permute_dims[dim] = 0;

  for (int i = dim + 1; i < out_dim; i++) {
    permute_dims[i] = i;
  }

  add_permute_node(graph, in, permute_dims, out);
}

void unsqueeze(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_unsqueeze_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.unsqueeze_copy.default, unsqueeze);
}

} // namespace vkcompute
