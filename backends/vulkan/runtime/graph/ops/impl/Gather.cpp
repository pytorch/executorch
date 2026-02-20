/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/utils/StorageUtils.h>

namespace vkcompute {

using utils::GPUMemoryLayout;
using utils::StorageType;

void resize_gather_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef index = args.at(1).refs.at(1);

  // Output shape is the same as index shape
  std::vector<int64_t> out_sizes = graph->sizes_of(index);
  graph->virtual_resize(out, out_sizes);
}

void add_gather_node(
    ComputeGraph& graph,
    const ValueRef input,
    const int64_t dim,
    const ValueRef index,
    const ValueRef out) {
  std::string kernel_name = "gather";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out), graph.meta_ubo(input), graph.meta_ubo(index)};

  const int64_t dim_whcn = graph.dim_of(input) - dim - 1;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{input, index}, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {static_cast<int32_t>(dim_whcn)},
      // Resize Args
      {},
      // Resizing Logic
      resize_gather_node));
}

void gather(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef input = args[0];
  ValueRef dim_ref = args[1];
  ValueRef index = args[2];
  ValueRef out = args[4];

  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);

  add_gather_node(graph, input, dim, index, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.gather.default, gather);
}

} // namespace vkcompute
