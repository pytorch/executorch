/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_repeat_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef repeats_ref = extra_args.at(0);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const std::vector<int64_t> repeats =
      graph->extract_int_or_symint_list(repeats_ref);

  const size_t out_ndim = std::max(in_sizes.size(), repeats.size());
  std::vector<int64_t> out_sizes(out_ndim);
  for (size_t i = 0; i < out_ndim; i++) {
    const size_t in_offset = i + in_sizes.size() - out_ndim;
    const size_t rep_offset = i + repeats.size() - out_ndim;
    // Prepend 1s to in_sizes if repeats is longer, and vice versa
    const int64_t in_size =
        (i >= out_ndim - in_sizes.size()) ? in_sizes[in_offset] : 1;
    const int64_t r =
        (i >= out_ndim - repeats.size()) ? repeats[rep_offset] : 1;
    out_sizes[i] = in_size * r;
  }
  graph->virtual_resize(out, out_sizes);
}

void add_repeat_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef repeats_ref,
    ValueRef out) {
  std::string kernel_name = "repeat";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      {graph.meta_ubo(out), graph.meta_ubo(in)},
      {},
      {},
      {repeats_ref},
      resize_repeat_node));
}

void repeat(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_repeat_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.repeat.default, repeat);
}

} // namespace vkcompute
