/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Reduce.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

void arg_reduce_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const std::string& op_name) {
  int arg_idx = 0;
  const ValueRef in = args.at(arg_idx++);
  const ValueRef dim = args.at(arg_idx++);
  const ValueRef keepdim = args.at(arg_idx++);
  const ValueRef out = args.at(arg_idx++);

  VK_CHECK_COND(graph.is_buffer_storage(in));

  int64_t dim_val = 0;
  if (graph.val_is_not_none(dim)) {
    dim_val = graph.extract_scalar<int64_t>(dim);
  }
  const int64_t ndim = graph.dim_of(in);
  const int64_t normalized_dim = normalize(dim_val, graph.dim_of(in));

  VK_CHECK_COND(normalized_dim == ndim - 1);

  // Use the reduce_per_row_node function
  add_reduce_per_row_node(graph, in, keepdim, out, op_name);
}

void argmin(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  arg_reduce_impl(graph, args, "argmin");
}

void argmax(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  arg_reduce_impl(graph, args, "argmax");
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.argmin.default, argmin);
  VK_REGISTER_OP(aten.argmax.default, argmax);
}

} // namespace vkcompute
