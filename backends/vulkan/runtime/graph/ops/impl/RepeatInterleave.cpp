/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/RepeatInterleave.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_repeat_interleave_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  const int64_t nrepeats = graph->extract_scalar<int64_t>(extra_args.at(0));
  int64_t repeat_dim = graph->extract_scalar<int64_t>(extra_args.at(1));

  std::vector<int64_t> new_sizes = graph->sizes_of(in);
  repeat_dim = normalize(repeat_dim, new_sizes.size());
  new_sizes.at(repeat_dim) *= nrepeats;

  graph->virtual_resize(out, new_sizes);
}

void add_repeat_interleave_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef num_repeats,
    const ValueRef dim,
    const ValueRef out) {
  const int32_t nrepeats = graph.extract_scalar<int32_t>(num_repeats);
  const int32_t repeat_dim =
      graph.extract_whcn_dim<int32_t>(dim, graph.dim_of(in));

  VK_CHECK_COND(repeat_dim != graph.packed_dim_of(out));
  VK_CHECK_COND(repeat_dim != graph.packed_dim_of(in));

  std::string kernel_name = "repeat_interleave";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      // Parameter buffers
      {graph.logical_limits_ubo(in)},
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(in),
       nrepeats,
       repeat_dim},
      // Resize Args
      {num_repeats, dim},
      // Resizing Logic
      resize_repeat_interleave_node));
}

void repeat_interleave(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int args_i = 0;
  const ValueRef in = args[args_i++];
  const ValueRef num_repeats = args[args_i++];
  const ValueRef dim = args[args_i++];
  const ValueRef output_size = args[args_i++];
  const ValueRef out = args[args_i++];

  // Output size is not used in the kernel
  (void)output_size;

  add_repeat_interleave_node(graph, in, num_repeats, dim, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.repeat_interleave.self_int, repeat_interleave);
}

} // namespace vkcompute
