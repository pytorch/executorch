/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_binary_op_args(
    const api::vTensor& self,
    const api::vTensor& other,
    const api::vTensor& out) {
  VK_CHECK_COND(check_same_packed_dim(self, other, out));
  std::vector<int64_t> broadcasted_sizes =
      calculate_broadcasted_output_size(self, other);
  VK_CHECK_COND(out.sizes() == broadcasted_sizes);
}

void resize_binary_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);

  // TODO(T183442143): Verify tensors are broadcastable.
  vTensorPtr self = graph->get_tensor(args[1].refs[0]);
  vTensorPtr other = graph->get_tensor(args[1].refs[1]);

  std::vector<int64_t> new_out_sizes =
      calculate_broadcasted_output_size(*self, *other);

  out->virtual_resize(new_out_sizes);
}

void add_binary_op_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const std::string& op_name) {
  ValueRef arg1 = prepack_standard_like(graph, in1, out, true);
  ValueRef arg2 = prepack_standard_like(graph, in2, out, true);

  vTensorPtr t_in1 = graph.get_tensor(arg1);
  vTensorPtr t_in2 = graph.get_tensor(arg2);
  vTensorPtr t_out = graph.get_tensor(out);

  check_binary_op_args(*t_in1, *t_in2, *t_out);

  float alpha_val = 1.0f;
  // String is checked since floor_div passes in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.val_is_string(alpha)) {
    alpha_val = graph.extract_scalar<float>(alpha);
  }

  const struct BinaryOpsParams {
    const utils::ivec2 broadcast_params;
    const float alpha_val;
  } binary_ops_params{create_broadcast_params(*t_in1, *t_in2), alpha_val};

  std::string kernel_name("binary_");
  kernel_name.reserve(kShaderNameReserve);
  kernel_name += op_name;
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {{arg1, arg2}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {},
      // Specialization Constants
      {t_out->hashed_layout(), t_in1->hashed_layout(), t_in2->hashed_layout()},
      // Resizing Logic
      resize_binary_op_node,
      {},
      {{graph.sizes_pc_of(out),
        graph.sizes_pc_of(arg1),
        graph.sizes_pc_of(arg2),
        PushConstantDataInfo(&binary_ops_params, sizeof(binary_ops_params))}}));
}

#define DEFINE_BINARY_OP_WITH_ALPHA_FN(op_name)                          \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph, args[0], args[1], args[2], args[3], #op_name);            \
  }

#define DEFINE_BINARY_OP_FN(op_name)                                     \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph, args[0], args[1], kDummyValueRef, args[2], #op_name);     \
  }

DEFINE_BINARY_OP_WITH_ALPHA_FN(add);
DEFINE_BINARY_OP_WITH_ALPHA_FN(sub);

// Floor div does not have an alpha, but a string argument (which is unused) is
// passed in at the same location as the alpha argument in other op.
DEFINE_BINARY_OP_WITH_ALPHA_FN(floor_divide);

DEFINE_BINARY_OP_FN(mul);
DEFINE_BINARY_OP_FN(div);
DEFINE_BINARY_OP_FN(pow);
DEFINE_BINARY_OP_FN(minimum);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.add.Tensor, add);
  VK_REGISTER_OP(aten.sub.Tensor, sub);
  VK_REGISTER_OP(aten.mul.Tensor, mul);
  VK_REGISTER_OP(aten.div.Tensor, div);
  VK_REGISTER_OP(aten.div.Tensor_mode, floor_divide);
  VK_REGISTER_OP(aten.pow.Tensor_Tensor, pow);
  VK_REGISTER_OP(aten.minimum.default, minimum);
}

} // namespace vkcompute
