/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <array>

namespace vkcompute {

void check_adamw_step_args(
    ComputeGraph& graph,
    const ValueRef param,
    const ValueRef m,
    const ValueRef v,
    const ValueRef grad) {
  VK_CHECK_COND(graph.dtype_of(param) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(m) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(v) == vkapi::kFloat);
  VK_CHECK_COND(graph.dtype_of(grad) == vkapi::kFloat);

  const int32_t numel = graph.numel_of(param);
  VK_CHECK_COND(graph.numel_of(m) == numel);
  VK_CHECK_COND(graph.numel_of(v) == numel);
  VK_CHECK_COND(graph.numel_of(grad) == numel);
}

void add_adamw_step_node(
    ComputeGraph& graph,
    const ValueRef param,
    const ValueRef m,
    const ValueRef v,
    const ValueRef grad,
    const ValueRef lr,
    const ValueRef beta1,
    const ValueRef beta2,
    const ValueRef eps,
    const ValueRef weight_decay,
    const ValueRef bias_correction1,
    const ValueRef bias_correction2) {
  check_adamw_step_args(graph, param, m, v, grad);

  const float bc1 = graph.extract_scalar<float>(bias_correction1);
  const float bc2 = graph.extract_scalar<float>(bias_correction2);
  VK_CHECK_COND(bc1 != 0.0f);
  VK_CHECK_COND(bc2 != 0.0f);

  // Split into <=16-byte push-constant entries (PushConstantData.h limit).
  const std::array<float, 4> pc0 = {
      graph.extract_scalar<float>(lr),
      graph.extract_scalar<float>(beta1),
      graph.extract_scalar<float>(beta2),
      graph.extract_scalar<float>(eps)};
  const std::array<float, 3> pc1 = {
      graph.extract_scalar<float>(weight_decay), bc1, bc2};

  std::string kernel_name("adamw_step");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(param));
  add_dtype_suffix(kernel_name, graph.dtype_of(param));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{param, vkapi::kReadWrite},
       {m, vkapi::kReadWrite},
       {v, vkapi::kReadWrite},
       {grad, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {graph.numel_pc_of(param),
       PushConstantDataInfo(pc0.data(), sizeof(pc0)),
       PushConstantDataInfo(pc1.data(), sizeof(pc1))},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void adamw_step(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef param = args[arg_idx++];
  const ValueRef m = args[arg_idx++];
  const ValueRef v = args[arg_idx++];
  const ValueRef grad = args[arg_idx++];
  const ValueRef lr = args[arg_idx++];
  const ValueRef beta1 = args[arg_idx++];
  const ValueRef beta2 = args[arg_idx++];
  const ValueRef eps = args[arg_idx++];
  const ValueRef weight_decay = args[arg_idx++];
  const ValueRef bias_correction1 = args[arg_idx++];
  const ValueRef bias_correction2 = args[arg_idx++];
  add_adamw_step_node(
      graph,
      param,
      m,
      v,
      grad,
      lr,
      beta1,
      beta2,
      eps,
      weight_decay,
      bias_correction1,
      bias_correction2);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.adamw_step.default, adamw_step);
}

} // namespace vkcompute
