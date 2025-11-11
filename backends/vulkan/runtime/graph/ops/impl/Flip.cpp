/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Custom global workgroup size function for flip
utils::uvec3 flip_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return graph->create_global_wg_size(out);
}

void check_flip_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(in) == WHCN::kChannelsDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kChannelsDim);
}

void resize_flip_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  graph->virtual_resize(out, graph->sizes_of(in));
}

utils::ivec4 create_whcn_bitmap(
    const std::vector<int64_t>& list,
    const int64_t ndim) {
  std::vector<int64_t> bm(4, 0);
  for (const auto e : list) {
    auto x = (e % ndim + ndim) % ndim; // normalize
    x = ndim - 1 - x; // reverse
    bm.at(x) = 1;
  }
  return utils::make_ivec4(bm);
}

void add_flip_node(
    ComputeGraph& graph,
    const ValueRef in,
    const std::vector<int64_t>& dim_list,
    const ValueRef out) {
  check_flip_args(graph, in, out);

  const auto dim_bitmap = create_whcn_bitmap(dim_list, graph.dim_of(in));

  std::string kernel_name("flip");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      flip_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {
          {out, vkapi::kWrite},
          {in, vkapi::kRead},
      },
      // Parameter buffers
      {
          graph.logical_limits_ubo(out),
          graph.sizes_ubo(out),
          graph.create_params_buffer(dim_bitmap),
      },
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_flip_node));
}

void flip(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef in = args[0];
  auto dims = graph.get_int_list(args[1]);
  ValueRef out = args[2];

  add_flip_node(graph, in, *dims, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.flip.default, flip);
}

} // namespace vkcompute
