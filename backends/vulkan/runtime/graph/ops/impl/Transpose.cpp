/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>

namespace vkcompute {

/*
 * Transposing for sizes and strides is as simple as swapping the values at
 * dim0 and dim1 in the sizes/strides vector.
 */
void swap_vector_inplace(
    std::vector<int64_t>& vec,
    const int64_t dim0,
    const int64_t dim1) {
  std::iter_swap(vec.begin() + dim0, vec.begin() + dim1);
}

/*
 * Transposing the dim order is a bit more unintuitive. dim0 and dim1 have
 * swapped their "identities", so we need to swap the values of dim0 and dim1
 * wherever they appear in the dim order vector. Compare this to just swapping
 * the elements at dim0 and dim1 in the strides or sizes vectors.
 */
void transpose_dim_order_inplace(
    std::vector<int64_t>& dim_order,
    const int64_t dim0,
    const int64_t dim1) {
  for (int i = 0; i < dim_order.size(); ++i) {
    if (dim_order[i] == dim0) {
      dim_order[i] = dim1;
    } else if (dim_order[i] == dim1) {
      dim_order[i] = dim0;
    }
  }
}

void resize_transpose_view_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;
  vTensorPtr out = graph->get_tensor(extra_args[0]);
  vTensorPtr in = graph->get_tensor(extra_args[1]);

  const int64_t dim0 = graph->extract_scalar<int64_t>(extra_args[2]);
  const int64_t dim1 = graph->extract_scalar<int64_t>(extra_args[3]);

  std::vector<int64_t> new_sizes = in->sizes();
  std::vector<int64_t> new_dim_order = in->dim_order();

  swap_vector_inplace(new_sizes, dim0, dim1);
  transpose_dim_order_inplace(new_dim_order, dim0, dim1);

  out->virtual_reconfigure(new_sizes, new_dim_order);
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

  std::vector<int64_t> new_sizes = graph.sizes_of(input_ref);
  std::vector<int64_t> new_dim_order = graph.dim_order_of(input_ref);

  swap_vector_inplace(new_sizes, dim0, dim1);
  transpose_dim_order_inplace(new_dim_order, dim0, dim1);

  graph.get_tensor(out_ref)->virtual_reconfigure(new_sizes, new_dim_order);

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
