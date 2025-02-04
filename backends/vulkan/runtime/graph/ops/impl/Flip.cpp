/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_flip_args(const api::vTensor& in, const api::vTensor& out) {
  VK_CHECK_COND(check_packed_dim_is(in, WHCN::kChannelsDim));
  VK_CHECK_COND(check_packed_dim_is(out, WHCN::kChannelsDim));
}

void resize_flip_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);

  out->virtual_resize(in->sizes());
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
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);
  check_flip_args(*t_in, *t_out);

  const auto dim_bitmap = create_whcn_bitmap(dim_list, t_in->dim());

  std::string kernel_name("flip");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
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
      // Specialization Constants
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
