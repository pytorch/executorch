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

//
// sym_size
//

void sym_size_impl(ComputeGraph* graph, const std::vector<ValueRef>& args) {
  const ValueRef in_tensor = args.at(0);
  const ValueRef dim = args.at(1);
  const ValueRef out_symint = args.at(2);

  const int64_t dim_val = graph->extract_scalar<int64_t>(dim);
  const int64_t size_at_dim = graph->size_at<int64_t>(dim_val, in_tensor);

  graph->set_symint(out_symint, static_cast<int32_t>(size_at_dim));
}

void resize_sym_size_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args; // Unused parameter
  sym_size_impl(graph, resize_args);
}

/*
 * This operator takes a tensor and an integer dimension as inputs, and produces
 * a symint as output. The symint's value is the size of the tensor at the
 * specified dimension.
 */
void sym_size_int(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  sym_size_impl(&graph, args);

  graph.execute_nodes().emplace_back(
      new ExecuteNode(resize_sym_size_node, args));
}

//
// binary operators
//

void sym_add_impl(ComputeGraph* graph, const std::vector<ValueRef>& args) {
  const ValueRef a = args.at(0);
  const ValueRef b = args.at(1);
  const ValueRef out = args.at(2);

  const int32_t a_val = graph->read_symint(a);
  const int32_t b_val = graph->read_symint(b);
  const int32_t result = a_val + b_val;

  graph->set_symint(out, result);
}

void resize_sym_add_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args; // Unused parameter
  sym_add_impl(graph, resize_args);
}

/*
 * This operator takes two symints as inputs and produces a symint as output.
 * The output symint's value is the sum of the two input symints.
 */
void sym_add(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  sym_add_impl(&graph, args);

  graph.execute_nodes().emplace_back(
      new ExecuteNode(resize_sym_add_node, args));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(sym_size.int, sym_size_int);
  VK_REGISTER_OP(add, sym_add);
}

} // namespace vkcompute
