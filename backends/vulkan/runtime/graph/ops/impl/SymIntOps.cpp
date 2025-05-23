/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

namespace vkcompute {

void resize_sym_size_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args; // Unused parameter

  ValueRef out_symint_ref = extra_args[0];
  ValueRef in_tensor_ref = extra_args[1];

  int64_t dim = graph->extract_scalar<int64_t>(extra_args[2]);
  int64_t size_at_dim = graph->size_at<int64_t>(dim, in_tensor_ref);

  graph->set_symint(out_symint_ref, static_cast<int32_t>(size_at_dim));
}

/*
 * This operator takes a tensor and an integer dimension as inputs, and produces
 * a symint as output. The symint's value is the size of the tensor at the
 * specified dimension.
 */
void sym_size_int(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef in_tensor = args[0];
  ValueRef dim = args[1];
  ValueRef out_symint = args[2];

  int64_t dim_val = graph.extract_scalar<int64_t>(dim);

  int64_t size_at_dim = graph.size_at<int64_t>(dim_val, in_tensor);
  graph.set_symint(out_symint, static_cast<int32_t>(size_at_dim));

  graph.execute_nodes().emplace_back(
      new ExecuteNode(resize_sym_size_node, {out_symint, in_tensor, dim}));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sym_size.int, sym_size_int);
}

} // namespace vkcompute
