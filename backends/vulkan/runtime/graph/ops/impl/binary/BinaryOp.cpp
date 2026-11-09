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

namespace vkcompute {

void check_binary_op_args(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef other,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(self) == graph.packed_dim_of(other));
  VK_CHECK_COND(graph.packed_dim_of(self) == graph.packed_dim_of(out));

  const std::vector<int64_t> self_sizes = graph.sizes_of(self);
  const std::vector<int64_t> other_sizes = graph.sizes_of(other);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);

  std::vector<int64_t> broadcasted_sizes =
      calculate_broadcasted_output_size(self_sizes, other_sizes);
  VK_CHECK_COND(out_sizes == broadcasted_sizes);
}

void resize_binary_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  // TODO(T183442143): Verify tensors are broadcastable.
  const ValueRef self = args.at(1).refs.at(0);
  const ValueRef other = args.at(1).refs.at(1);

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  const std::vector<int64_t> other_sizes = graph->sizes_of(other);
  const std::vector<int64_t> new_out_sizes =
      calculate_broadcasted_output_size(self_sizes, other_sizes);

  graph->virtual_resize(out, new_out_sizes);
}

void add_binary_op_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const BinaryOpType op_type) {
  ValueRef arg1 = prepack_standard_like(graph, in1, out, true);
  ValueRef arg2 = prepack_standard_like(graph, in2, out, true);

  check_binary_op_args(graph, arg1, arg2, out);
  if (!graph.is_buffer_storage(out)) {
    VK_CHECK_COND(graph.has_standard_axis_map(arg1));
    VK_CHECK_COND(graph.has_standard_axis_map(arg2));
    VK_CHECK_COND(graph.has_standard_axis_map(out));
  }

  float alpha_val = 1.0f;
  if (is_valid(alpha)) {
    alpha_val = graph.extract_scalar<float>(alpha);
  }

  const bool is_comparison =
      op_type >= BinaryOpType::kEq && op_type <= BinaryOpType::kGe;
  std::string kernel_name(is_comparison ? "binary_cmp" : "binary_op");
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(in1));

  vkapi::ParamsBindList ubos = {
      graph.meta_ubo(out), graph.meta_ubo(arg1), graph.meta_ubo(arg2)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_vec4_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{arg1, arg2}, vkapi::kRead}},
      // Shader params buffers
      ubos,
      // Push Constants
      {{PushConstantDataInfo(&alpha_val, sizeof(float))}},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(arg1),
       graph.hashed_layout_of(arg2),
       static_cast<int32_t>(op_type)},
      // Resize Args
      {},
      // Resizing Logic
      resize_binary_op_node));
}

#define DEFINE_BINARY_OP_WITH_ALPHA_FN(op_name, op_type)                   \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) {   \
    return add_binary_op_node(                                             \
        graph, args[0], args[1], args[2], args[3], BinaryOpType::op_type); \
  }

#define DEFINE_BINARY_OP_FN(op_name, op_type)                            \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph,                                                           \
        args[0],                                                         \
        args[1],                                                         \
        kDummyValueRef,                                                  \
        args[2],                                                         \
        BinaryOpType::op_type);                                          \
  }

DEFINE_BINARY_OP_WITH_ALPHA_FN(add, kAdd);
DEFINE_BINARY_OP_WITH_ALPHA_FN(sub, kSub);

void floor_divide(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // div.Tensor_mode is implemented for rounding_mode="floor" only.
  VK_CHECK_COND(
      graph.val_is_string(args[2]) && graph.extract_string(args[2]) == "floor");
  return add_binary_op_node(
      graph,
      args[0],
      args[1],
      kDummyValueRef,
      args[3],
      BinaryOpType::kFloorDivide);
}

DEFINE_BINARY_OP_FN(mul, kMul);
DEFINE_BINARY_OP_FN(swiglu, kSwiglu);
DEFINE_BINARY_OP_FN(div, kDiv);
DEFINE_BINARY_OP_FN(pow, kPow);
DEFINE_BINARY_OP_FN(minimum, kMinimum);
DEFINE_BINARY_OP_FN(eq, kEq);
DEFINE_BINARY_OP_FN(lt, kLt);
DEFINE_BINARY_OP_FN(le, kLe);
DEFINE_BINARY_OP_FN(gt, kGt);
DEFINE_BINARY_OP_FN(ge, kGe);
DEFINE_BINARY_OP_FN(bitwise_and, kBitwiseAnd);
DEFINE_BINARY_OP_FN(bitwise_or, kBitwiseOr);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.add.Tensor, add);
  VK_REGISTER_OP(aten.sub.Tensor, sub);
  VK_REGISTER_OP(aten.mul.Tensor, mul);
  VK_REGISTER_OP(et_vk.swiglu.default, swiglu);
  VK_REGISTER_OP(aten.div.Tensor, div);
  VK_REGISTER_OP(aten.div.Tensor_mode, floor_divide);
  VK_REGISTER_OP(aten.pow.Tensor_Tensor, pow);
  VK_REGISTER_OP(aten.minimum.default, minimum);
  VK_REGISTER_OP(aten.eq.Tensor, eq);
  VK_REGISTER_OP(aten.lt.Tensor, lt);
  VK_REGISTER_OP(aten.le.Tensor, le);
  VK_REGISTER_OP(aten.gt.Tensor, gt);
  VK_REGISTER_OP(aten.ge.Tensor, ge);
  VK_REGISTER_OP(aten.bitwise_and.Tensor, bitwise_and);
  VK_REGISTER_OP(aten.logical_and.default, bitwise_and);
  VK_REGISTER_OP(aten.bitwise_or.Tensor, bitwise_or);
  VK_REGISTER_OP(aten.logical_or.default, bitwise_or);
}

} // namespace vkcompute
