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

namespace at {
namespace native {
namespace vulkan {

std::string get_arithmetic_shader_name(const std::string& op_name) {
  return "arithmetic_" + op_name;
}

void resize_arithmetic_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensor& out = graph->get_val(args[0].refs[0]).toTensor();
  vTensor& self = graph->get_val(args[1].refs[0]).toTensor();
  vTensor& other = graph->get_val(args[1].refs[1]).toTensor();

  std::vector<int64_t> new_out_sizes(
      std::max(self.sizes().size(), other.sizes().size()));

  for (int i = -1; i >= -new_out_sizes.size(); --i) {
    new_out_sizes[new_out_sizes.size() + i] = std::max(
        api::utils::val_at(i, self.sizes()),
        api::utils::val_at(i, other.sizes()));
  }

  out.virtual_resize(new_out_sizes);
}

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const std::string& op_name) {
  ValueRef arg1 = prepack_if_tensor_ref(graph, in1);
  ValueRef arg2 = prepack_if_tensor_ref(graph, in2);

  vTensor& t_in1 = graph.get_val(arg1).toTensor();
  vTensor& t_in2 = graph.get_val(arg2).toTensor();
  vTensor& t_out = graph.get_val(out).toTensor();

  api::utils::uvec3 global_size = t_out.virtual_extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  float alpha_val = 1.0f;
  // String is checked since floor_div passes in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.get_val(alpha).isString()) {
    alpha_val = extract_scalar<float>(graph.get_val(alpha));
  }

  std::stringstream kernel_name;
  kernel_name << "binary_" << op_name;
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
       graph.create_params_buffer(alpha_val)},
      // Resizing
      {alpha},
      resize_arithmetic_node));
}

#define DEFINE_ARITHMETIC_WITH_ALPHA_FN(function, shader)                 \
  void function(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_arithmetic_node(                                           \
        graph, args[0], args[1], args[2], args[3], #shader);              \
  }

#define DEFINE_ARITHMETIC_FN(function, shader)                            \
  void function(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_arithmetic_node(                                           \
        graph, args[0], args[1], kDummyValueRef, args[2], #shader);       \
  }

DEFINE_ARITHMETIC_WITH_ALPHA_FN(add, add);
DEFINE_ARITHMETIC_WITH_ALPHA_FN(sub, sub);

// Floor div does not have an alpha, but a string argument (which is unused) is
// passed in at the same location as the alpha argument in other op.
DEFINE_ARITHMETIC_WITH_ALPHA_FN(floor_div, floor_divide);

DEFINE_ARITHMETIC_FN(mul, mul);
DEFINE_ARITHMETIC_FN(div, div);
DEFINE_ARITHMETIC_FN(pow, pow);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.add.Tensor, add);
  VK_REGISTER_OP(aten.sub.Tensor, sub);
  VK_REGISTER_OP(aten.mul.Tensor, mul);
  VK_REGISTER_OP(aten.div.Tensor, div);
  VK_REGISTER_OP(aten.div.Tensor_mode, floor_div);
  VK_REGISTER_OP(aten.pow.Tensor_Tensor, pow);
}

} // namespace vulkan
} // namespace native
} // namespace at
