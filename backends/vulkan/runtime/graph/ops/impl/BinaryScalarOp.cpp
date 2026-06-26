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

#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>

#include <vector>

namespace vkcompute {

namespace {

vkapi::ScalarType resolve_scalar_extract_dtype(
    ComputeGraph& graph,
    const ValueRef scalar,
    const vkapi::ScalarType tensor_dtype) {
  // For float tensors, ensure that the scalar argument is extracted as a float
  // to avoid having to generate additional shader variants for float/half
  // tensor + int scalar.
  if (tensor_dtype == vkapi::kFloat || tensor_dtype == vkapi::kHalf) {
    return vkapi::kFloat;
  }
  if (graph.val_is_bool(scalar) || graph.val_is_symint(scalar)) {
    return vkapi::kInt;
  }
  return graph.dtype_of(scalar);
}

int32_t extract_int32_scalar(ComputeGraph& graph, const ValueRef scalar) {
  if (graph.val_is_int(scalar)) {
    return utils::safe_downcast<int32_t>(graph.get_int(scalar));
  }
  if (graph.val_is_bool(scalar)) {
    return graph.get_bool(scalar) ? 1 : 0;
  }
  if (graph.val_is_symint(scalar)) {
    return graph.read_symint(scalar);
  }
  VK_THROW(
      "Expected int, bool, or SymInt scalar, got: ",
      graph.get_val_type(scalar));
}

} // namespace

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

  const vkapi::ScalarType tensor_dtype = graph.dtype_of(in);
  const vkapi::ScalarType scalar_dtype =
      resolve_scalar_extract_dtype(graph, scalar, tensor_dtype);
  std::vector<PushConstantDataInfo> push_constants;
  if (scalar_dtype == vkapi::kInt) {
    const int32_t scalar_val = extract_int32_scalar(graph, scalar);
    push_constants.emplace_back(&scalar_val, sizeof(scalar_val));
  } else if (scalar_dtype == vkapi::kFloat) {
    const float scalar_val = graph.extract_scalar<float>(scalar);
    push_constants.emplace_back(&scalar_val, sizeof(scalar_val));
  } else {
    VK_THROW("Unsupported tensor-scalar op scalar dtype: ", scalar_dtype);
  }

  std::string kernel_name = op_name + "_scalar";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, tensor_dtype);
  add_dtype_suffix(kernel_name, scalar_dtype);

  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out), graph.meta_ubo(in)};

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
      push_constants,
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

void eq_tensor_scalar(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_binary_scalar_op_node(graph, args[0], args[1], args[2], "eq");
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.pow.Tensor_Scalar, pow_tensor_scalar);
  VK_REGISTER_OP(aten.eq.Scalar, eq_tensor_scalar);
}

} // namespace vkcompute
