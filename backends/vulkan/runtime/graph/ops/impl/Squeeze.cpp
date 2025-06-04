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

void squeeze_copy_dims(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_squeeze_copy_dims_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.squeeze_copy.dims, squeeze_copy_dims);
}

} // namespace vkcompute
