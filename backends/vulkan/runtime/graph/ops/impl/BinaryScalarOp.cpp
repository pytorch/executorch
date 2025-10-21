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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_binary_scalar_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);

  graph->virtual_resize(out, in_sizes);
}

void add_binary_scalar_op_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef scalar,
    const ValueRef out,
    const std::string& op_name) {
  ValueRef arg = prepack_standard_like(graph, in, out, true);

  // Extract scalar value
  float scalar_val = graph.extract_scalar<float>(scalar);

  // Pick shader
  std::string kernel_name = op_name + "_scalar";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(in));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out),
      graph.meta_ubo(in),
      graph.create_params_buffer(scalar_val)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {arg, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_binary_scalar_op_node));
}

void pow_tensor_scalar(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_binary_scalar_op_node(graph, args[0], args[1], args[2], "pow");
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.pow.Tensor_Scalar, pow_tensor_scalar);
}

} // namespace vkcompute
