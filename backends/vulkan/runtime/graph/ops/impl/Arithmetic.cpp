/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Arithmetic.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OpUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

namespace at {
namespace native {
namespace vulkan {

#define DEFINE_ARITHMETIC_WITH_ALPHA_FN(function, shader)                 \
  void function(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_arithmetic_node(                                           \
        graph, args[0], args[1], args[2], args[3], VK_KERNEL(shader));    \
  }

#define DEFINE_ARITHMETIC_FN(function, shader)                                \
  void function(ComputeGraph& graph, const std::vector<ValueRef>& args) {     \
    return add_arithmetic_node(                                               \
        graph, args[0], args[1], kDummyValueRef, args[2], VK_KERNEL(shader)); \
  }

DEFINE_ARITHMETIC_WITH_ALPHA_FN(add, add);
DEFINE_ARITHMETIC_WITH_ALPHA_FN(sub, sub);

// Floor div does not have an alpha, but a string argument (which is unused) is
// passed in at the same location as the alpha argument in other op.
DEFINE_ARITHMETIC_WITH_ALPHA_FN(floor_div, floor_divide);

DEFINE_ARITHMETIC_FN(mul, mul);
DEFINE_ARITHMETIC_FN(div, div);
DEFINE_ARITHMETIC_FN(pow, pow);

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const api::ShaderInfo& shader) {
  ValueRef arg1 = prepack_if_tensor_ref(graph, in1);
  ValueRef arg2 = prepack_if_tensor_ref(graph, in2);

  vTensor& t_in1 = graph.get_val(arg1).toTensor();
  vTensor& t_in2 = graph.get_val(arg2).toTensor();
  vTensor& t_out = graph.get_val(out).toTensor();

  api::utils::uvec3 global_size = t_out.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  float alpha_val = 1.0f;
  // String is checked since floor_div passes in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.get_val(alpha).isString()) {
    alpha_val = extract_scalar<float>(graph.get_val(alpha));
  }

  ArithmeticParams block{
      get_size_as_ivec4(t_out),
      get_size_as_ivec4(t_in1),
      get_size_as_ivec4(t_in2),
      alpha_val,
  };
  api::UniformParamsBuffer params(graph.context(), block);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      global_size,
      local_size,
      {{out, api::MemoryAccessType::WRITE},
       {{arg1, arg2}, api::MemoryAccessType::READ}},
      std::move(params)));
}

} // namespace vulkan
} // namespace native
} // namespace at
