/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Transpose.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>

namespace vkcompute {

void resize_transpose_view_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;
  const ValueRef out = extra_args.at(0);
  const ValueRef in = extra_args.at(1);

  const int64_t dim0 = graph->extract_scalar<int64_t>(extra_args.at(2));
  const int64_t dim1 = graph->extract_scalar<int64_t>(extra_args.at(3));

  std::vector<int64_t> new_sizes = graph->sizes_of(in);
  // Transpose the resized input sizes
  std::iter_swap(new_sizes.begin() + dim0, new_sizes.begin() + dim1);
  graph->virtual_resize(out, new_sizes);
}

void check_transpose_view_args(
    ComputeGraph& graph,
    ValueRef in_ref,
    const int64_t dim0,
    const int64_t dim1,
    ValueRef out_ref) {
  VK_CHECK_COND(
      graph.val_is_view_of(out_ref, in_ref),
      "output tensor must be a view of the input tensor");

  const int64_t in_ndim = graph.dim_of(in_ref);
  VK_CHECK_COND(
      dim0 >= 0 && dim0 < in_ndim, "dim0 is not in the range of [0, in_ndim)");
  VK_CHECK_COND(
      dim1 >= 0 && dim1 < in_ndim, "dim1 is not in the range of [0, in_ndim)");
}

void add_transpose_view_node(
    ComputeGraph& graph,
    ValueRef input_ref,
    ValueRef dim0_ref,
    ValueRef dim1_ref,
    ValueRef out_ref) {
  const int64_t dim0 = graph.extract_scalar<int64_t>(dim0_ref);
  const int64_t dim1 = graph.extract_scalar<int64_t>(dim1_ref);

  check_transpose_view_args(graph, input_ref, dim0, dim1, out_ref);
  graph.virtual_clone(out_ref, input_ref);
  graph.virtual_transpose(out_ref, dim0, dim1);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      resize_transpose_view_node, {out_ref, input_ref, dim0_ref, dim1_ref}));
}

void transpose(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef out = args[3];
  return add_transpose_view_node(
      graph,
      args[0], // input
      args[1], // dim0
      args[2], // dim1
      out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.transpose.int, transpose);
}

} // namespace vkcompute
