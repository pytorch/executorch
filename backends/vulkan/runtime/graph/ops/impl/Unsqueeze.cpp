/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Permute.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/View.h>
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
  if (dim < 0) {
    dim += out_dim;
  }

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

void resize_unsqueeze_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef dims_ref = extra_args.at(0);

  std::vector<int64_t> dims_vec;
  if (graph->is_scalar_or_none(dims_ref)) {
    // Handle scalar case
    int64_t dim = graph->extract_scalar<int64_t>(dims_ref);
    dims_vec.push_back(dim);
  } else {
    // Handle list case
    const IntListPtr dims = graph->get_int_list(dims_ref);
    dims_vec.assign(dims->begin(), dims->end());
  }

  std::vector<int64_t> out_sizes = graph->sizes_of(in);

  // Insert singleton dimensions at the specified positions
  for (auto dim : dims_vec) {
    int64_t d = dim;
    if (d < 0) {
      d += static_cast<int64_t>(out_sizes.size()) + 1;
    }
    out_sizes.insert(out_sizes.begin() + d, 1);
  }

  graph->virtual_resize(out, out_sizes);
}

void unsqueeze(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef in = args.at(idx++);
  const ValueRef dims = args.at(idx++);
  const ValueRef out = args.at(idx++);

  std::vector<ValueRef> resize_args = {dims};
  if (graph.is_buffer_storage(in)) {
    return add_view_copy_buffer_node(
        graph, in, out, resize_args, resize_unsqueeze_node);
  }
  return add_unsqueeze_node(graph, in, dims, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.unsqueeze_copy.default, unsqueeze);
}

} // namespace vkcompute
