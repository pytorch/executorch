/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

struct GridPriorsParam final {
  int32_t stride;
  float offset;
};

void resize_grid_priors_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = extra_args.at(0);
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const int64_t height = in_sizes.at(in_sizes.size() - 2);
  const int64_t width = in_sizes.at(in_sizes.size() - 1);
  const std::vector<int64_t> sizes = {height * width, 2};
  graph->virtual_resize(out, sizes);
}

void add_grid_priors_node(
    ComputeGraph& graph,
    const ValueRef& in,
    const ValueRef& stride_ref,
    const ValueRef& offset_ref,
    const ValueRef& out) {
  const int32_t stride = graph.extract_scalar<int32_t>(stride_ref);
  const float offset = graph.extract_scalar<float>(offset_ref);

  std::string kernel_name = "grid_priors";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  const GridPriorsParam param = {stride, offset};
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {
          {out, vkapi::kWrite},
      },
      // Shader params buffers
      {
          graph.sizes_ubo(in),
          graph.sizes_ubo(out),
          graph.create_params_buffer(param),
      },
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {in},
      // Resizing Logic
      resize_grid_priors_node));
}

void grid_priors(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_grid_priors_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.grid_priors.default, grid_priors);
}
} // namespace vkcompute
