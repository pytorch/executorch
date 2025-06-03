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
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef out) {
  const int64_t in_dim = graph.dim_of(in);
  const int64_t out_dim = graph.dim_of(out);

  VK_CHECK_COND(
      in_dim < 4, "Cannot unsqueeze a tensor with more than 3 dimensions");

  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);

  std::vector<int64_t> permute_dims(out_dim);
  for (int i = 1; i <= dim; i++) {
    permute_dims[i - 1] = i;
  }
  permute_dims[dim] = 0;

  for (int i = dim + 1; i < out_dim; i++) {
    permute_dims[i] = i;
  }

  const ValueRef permute_dims_ref =
      graph.add_scalar_list<int64_t>(std::vector<int64_t>(permute_dims));
  add_permute_node(graph, in, permute_dims_ref, out);
}

void unsqueeze(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_unsqueeze_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.unsqueeze_copy.default, unsqueeze);
}

} // namespace vkcompute
