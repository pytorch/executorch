/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Clone.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Permute.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/View.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_squeeze_copy_dims_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dims_ref,
    const ValueRef out) {
  const int64_t in_dim = graph.dim_of(in);
  const std::vector<int64_t> in_sizes = graph.sizes_of(in);
  const std::vector<int64_t> out_sizes = graph.sizes_of(in);

  const std::vector<int64_t> dims = graph.extract_int_or_symint_list(dims_ref);
  std::vector<int64_t> squeeze_dims;
  // Filter out edge cases that we don't need squeeze:
  // 1. The size of squeeze dim is larger than 1.
  // 2. Squeeze outter most dim
  // For these cases, just pass input to output via clone.
  for (int i = 0; i < dims.size(); ++i) {
    if (dims.at(i) != 0 && in_sizes.at(dims.at(i)) == 1) {
      squeeze_dims.push_back(dims.at(i));
    }
  }
  if (squeeze_dims.size() == 0) {
    add_clone_node(graph, in, out);
  } else {
    std::vector<int64_t> permute_dims(in_dim);
    for (int i = 0; i < in_dim; ++i) {
      permute_dims.at(i) = i;
    }
    for (auto& elem : squeeze_dims) {
      auto it = std::find(permute_dims.begin(), permute_dims.end(), elem);
      VK_CHECK_COND(
          it != permute_dims.end(), "Squeeze dim not found in permute_dims");
      std::rotate(permute_dims.begin(), it, it + 1);
    }

    const ValueRef permute_dims_ref =
        graph.add_scalar_list<int64_t>(std::vector<int64_t>(permute_dims));
    add_permute_node(graph, in, permute_dims_ref, out);
  }
}

void resize_squeeze_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef dims_ref = extra_args.at(0);

  const IntListPtr dims = graph->get_int_list(dims_ref);

  std::vector<int64_t> out_sizes = graph->sizes_of(in);

  // Remove the dimensions specified in dims if their size is 1
  for (int64_t dim : *dims) {
    if (dim >= 0 && dim < static_cast<int64_t>(out_sizes.size()) &&
        out_sizes[dim] == 1) {
      out_sizes.erase(out_sizes.begin() + dim);
      // After erasing, all subsequent dims shift left by one
      // So we need to decrement all subsequent dims in dims
      for (auto& d : *dims) {
        if (d > dim) {
          --d;
        }
      }
    }
  }

  graph->virtual_resize(out, out_sizes);
}

void squeeze_copy_dims(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef in = args.at(idx++);
  const ValueRef dims = args.at(idx++);
  const ValueRef out = args.at(idx++);

  std::vector<ValueRef> resize_args = {dims};

  if (graph.is_buffer_storage(in)) {
    return add_view_copy_buffer_node(
        graph, in, out, resize_args, resize_squeeze_node);
  }
  return add_squeeze_copy_dims_node(graph, in, dims, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.squeeze_copy.dims, squeeze_copy_dims);
}

} // namespace vkcompute
