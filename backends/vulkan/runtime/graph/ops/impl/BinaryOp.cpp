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
    const vTensor& self,
    const vTensor& other,
    const vTensor& out) {
  VK_CHECK_COND(check_same_memory_layout(self, other, out));
  std::vector<int64_t> broadcasted_sizes =
      calculate_broadcasted_output_size(self, other);
  VK_CHECK_COND(out.sizes() == broadcasted_sizes);
}

void resize_binary_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensor& out = graph->get_val(args[0].refs[0]).toTensor();

  // TODO(T183442143): Verify tensors are broadcastable.
  vTensor& self = graph->get_val(args[1].refs[0]).toTensor();
  vTensor& other = graph->get_val(args[1].refs[1]).toTensor();

  std::vector<int64_t> new_out_sizes =
      calculate_broadcasted_output_size(self, other);

  out.virtual_resize(new_out_sizes);
}

void add_binary_op_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const std::string& op_name) {
  ValueRef arg1 = prepack_if_tensor_ref(graph, in1);
  ValueRef arg2 =
      prepack_if_tensor_ref(graph, in2, graph.memory_layout_of(arg1));

  vTensor& t_in1 = graph.get_val(arg1).toTensor();
  vTensor& t_in2 = graph.get_val(arg2).toTensor();

  vTensor& t_out = graph.get_val(out).toTensor();

  check_binary_op_args(t_in1, t_in2, t_out);

  api::utils::uvec3 global_size = t_out.virtual_extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  float alpha_val = 1.0f;
  // String is checked since floor_div passes in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.get_val(alpha).isString()) {
    alpha_val = extract_scalar<float>(graph.get_val(alpha));
  }

  const api::utils::ivec2 broadcast_params =
      create_broadcast_params(t_in1, t_in2);

  std::stringstream kernel_name;
  kernel_name << "binary_" << op_name;
  apply_memory_layout_suffix(kernel_name, t_out);
  apply_dtype_suffix(kernel_name, t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name.str()),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE},
       {{arg1, arg2}, api::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out.gpu_sizes_ubo(),
       t_in1.gpu_sizes_ubo(),
       t_in2.gpu_sizes_ubo(),
       graph.create_params_buffer(broadcast_params),
       graph.create_params_buffer(alpha_val)},
      // Resizing
      resize_binary_op_node));
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

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.add.Tensor, add);
  VK_REGISTER_OP(aten.sub.Tensor, sub);
  VK_REGISTER_OP(aten.mul.Tensor, mul);
  VK_REGISTER_OP(aten.div.Tensor, div);
  VK_REGISTER_OP(aten.div.Tensor_mode, floor_divide);
  VK_REGISTER_OP(aten.pow.Tensor_Tensor, pow);
}

} // namespace vkcompute
