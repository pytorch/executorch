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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/binary/BinaryOp.h>

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
    const BinaryOpType op_type) {
  ValueRef arg = prepack_standard_like(graph, in, out, true);
  VK_CHECK_COND(check_same_packed_dim(graph, arg, out));
  if (!graph.is_buffer_storage(out)) {
    VK_CHECK_COND(graph.has_standard_axis_map(arg));
    VK_CHECK_COND(graph.has_standard_axis_map(out));
  }

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

  const bool is_comparison =
      op_type != BinaryOpType::kMul && op_type != BinaryOpType::kPow;
  std::string kernel_name(
      is_comparison ? "binary_scalar_cmp" : "binary_scalar_op");
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, tensor_dtype);
  add_dtype_suffix(kernel_name, scalar_dtype);

  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out), graph.meta_ubo(in)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_vec4_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {arg, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {int32_t(graph.original_dtype_of(out) == vkapi::kHalf),
       graph.hashed_layout_of(out),
       static_cast<int32_t>(op_type)},
      // Resize Args
      {},
      // Resizing Logic
      resize_binary_scalar_op_node));
}

#define DEFINE_BINARY_SCALAR_OP_FN(op_name, op_type)                     \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_scalar_op_node(                                    \
        graph, args[0], args[1], args[2], BinaryOpType::op_type);        \
  }

DEFINE_BINARY_SCALAR_OP_FN(pow_tensor_scalar, kPow);
DEFINE_BINARY_SCALAR_OP_FN(mul_tensor_scalar, kMul);
DEFINE_BINARY_SCALAR_OP_FN(eq_tensor_scalar, kEq);
DEFINE_BINARY_SCALAR_OP_FN(ne_tensor_scalar, kNe);
DEFINE_BINARY_SCALAR_OP_FN(lt_tensor_scalar, kLt);
DEFINE_BINARY_SCALAR_OP_FN(le_tensor_scalar, kLe);
DEFINE_BINARY_SCALAR_OP_FN(gt_tensor_scalar, kGt);
DEFINE_BINARY_SCALAR_OP_FN(ge_tensor_scalar, kGe);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.pow.Tensor_Scalar, pow_tensor_scalar);
  VK_REGISTER_OP(aten.mul.Scalar, mul_tensor_scalar);
  VK_REGISTER_OP(aten.eq.Scalar, eq_tensor_scalar);
  VK_REGISTER_OP(aten.ne.Scalar, ne_tensor_scalar);
  VK_REGISTER_OP(aten.lt.Scalar, lt_tensor_scalar);
  VK_REGISTER_OP(aten.le.Scalar, le_tensor_scalar);
  VK_REGISTER_OP(aten.gt.Scalar, gt_tensor_scalar);
  VK_REGISTER_OP(aten.ge.Scalar, ge_tensor_scalar);
}

} // namespace vkcompute
