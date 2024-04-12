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

constexpr float kDummyFloat = -1.0f;
const std::string kClampShaderName = "clamp";

void resize_unary_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr self = graph->get_tensor(args[1].refs[0]);

  out->virtual_resize(self->sizes());
}

void add_unary_op_node(
    ComputeGraph& graph,
    const ValueRef in,
    const float min,
    const float max,
    const ValueRef out,
    const std::string& op_name) {
  ValueRef arg = prepack_if_tensor_ref(graph, in);

  vTensorPtr t_out = graph.get_tensor(out);
  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::string kernel_name(op_name);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}, {arg, api::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->extents_ubo(),
       graph.create_params_buffer(min),
       graph.create_params_buffer(max)},
      // Resizing
      resize_unary_op_node));
}

float get_val_or_inf(ComputeGraph& graph, const ValueRef& val, bool max) {
  if (!graph.val_is_none(val)) {
    return graph.extract_scalar<float>(val);
  }
  return max ? std::numeric_limits<float>::infinity()
             : -std::numeric_limits<float>::infinity();
}

#define DEFINE_ACTIVATION_FN(op_name)                                    \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph, args[0], kDummyFloat, kDummyFloat, args[1], #op_name);    \
  }

#define DEFINE_CLAMP_FN(op_name)                                         \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*max = */ false),                \
        get_val_or_inf(graph, args[2], /*max = */ true),                 \
        args[3],                                                         \
        kClampShaderName);                                               \
  }

#define DEFINE_RELU_FN(op_name)                                          \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        0,                                                               \
        std::numeric_limits<float>::infinity(),                          \
        args[1],                                                         \
        kClampShaderName);                                               \
  }

DEFINE_ACTIVATION_FN(abs);
DEFINE_ACTIVATION_FN(sigmoid);
DEFINE_ACTIVATION_FN(tanh);
DEFINE_CLAMP_FN(clamp);
DEFINE_CLAMP_FN(hardtanh);
DEFINE_RELU_FN(relu);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.abs.default, abs);
  VK_REGISTER_OP(aten.clamp.default, clamp);
  VK_REGISTER_OP(aten.hardtanh.default, hardtanh);
  VK_REGISTER_OP(aten.relu.default, relu);
  VK_REGISTER_OP(aten.sigmoid.default, sigmoid);
  VK_REGISTER_OP(aten.tanh.default, tanh);
}

} // namespace vkcompute
