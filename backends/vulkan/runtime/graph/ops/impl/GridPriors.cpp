/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

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
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(extra_args[0]);
  std::vector<int64_t> in_sizes = in->sizes();
  int64_t height = in_sizes.at(in_sizes.size() - 2);
  int64_t width = in_sizes.at(in_sizes.size() - 1);
  std::vector<int64_t> sizes = {height * width, 2};
  out->virtual_resize(sizes);
}

void add_grid_priors_node(
    ComputeGraph& graph,
    const ValueRef& in,
    const ValueRef& stride_ref,
    const ValueRef& offset_ref,
    const ValueRef& out) {
  vTensorPtr t_out = graph.get_tensor(out);
  vTensorPtr t_in = graph.get_tensor(in);
  int32_t stride = graph.extract_scalar<int32_t>(stride_ref);
  float offset = graph.extract_scalar<float>(offset_ref);

  std::string kernel_name = "grid_priors";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  GridPriorsParam param = {stride, offset};
  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {
          {out, vkapi::MemoryAccessType::WRITE},
      },
      // Shader params buffers
      {
          t_in->sizes_ubo(),
          t_out->sizes_ubo(),
          graph.create_params_buffer(param),
      },
      // Specialization Constants
      {},
      resize_grid_priors_node,
      {in}));
}

void grid_priors(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_grid_priors_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.grid_priors.default, grid_priors);
}
} // namespace vkcompute
