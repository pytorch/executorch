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

void scalar_tensor(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef scalar_in = args.at(0);
  const ValueRef out = args.back();

  std::string kernel_name("scalar_tensor");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  const bool scalar_is_integer =
      graph.val_is_int(scalar_in) || graph.val_is_symint(scalar_in);
  add_dtype_suffix(
      kernel_name, scalar_is_integer ? vkapi::kInt : graph.dtype_of(scalar_in));

  vkapi::BufferBindInfo scalar_param;
  if (scalar_is_integer) {
    scalar_param = graph.get_or_create_int_param_buffer(scalar_in);
  } else {
    scalar_param =
        graph.create_params_buffer(graph.extract_scalar<float>(scalar_in));
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_gwg(out),
      graph.create_lwg(out),
      // Inputs and Outputs
      {{out, vkapi::kWrite}},
      // Shader params buffers
      {scalar_param},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.scalar_tensor.default, scalar_tensor);
  VK_REGISTER_OP(scalar_tensor.default, scalar_tensor);
}

} // namespace vkcompute
