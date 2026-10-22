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
  // Extract the scalar value from the first argument
  ValueRef scalar_in = args[0];

  // Get the output tensor reference
  ValueRef out = args[args.size() - 1];
  const vkapi::ScalarType scalar_dtype =
      graph.dtype_of(out) == vkapi::kInt ? vkapi::kInt : vkapi::kFloat;
  const vkapi::BufferBindInfo scalar_buffer = scalar_dtype == vkapi::kInt
      ? graph.create_params_buffer(graph.extract_scalar<int32_t>(scalar_in))
      : graph.create_params_buffer(graph.extract_scalar<float>(scalar_in));

  std::string kernel_name("scalar_tensor");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, scalar_dtype);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_gwg(out),
      graph.create_lwg(out),
      // Inputs and Outputs
      {{out, vkapi::kWrite}},
      // Shader params buffers
      {scalar_buffer},
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
}

} // namespace vkcompute
