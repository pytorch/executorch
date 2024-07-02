/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_clone_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "clone";
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      {t_out->texture_limits_ubo()}));
}

void clone(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // The vulkan delegate does not support changing memory format.
  return add_clone_node(graph, args[0], args[2]);
}

// Clone node is not the most efficient implementation for the aten.clone
// operation. A more efficient implementation can be achieved during vulkan
// export with the use of shared object. This clone node is introduced to enable
// a "copy" mechanism if there is no alternative (e.g. during direct
// ComputeGraph manipulation, we need to make a copy of a Tensor).

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.clone.default, clone);
}

} // namespace vkcompute
