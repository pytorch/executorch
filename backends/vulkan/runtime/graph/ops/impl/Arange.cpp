/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_arange_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);

  int start_val = 0;
  int step_val = 1;
  if (!graph->val_is_none(extra_args.at(0))) {
    start_val = graph->extract_scalar<int64_t>(extra_args.at(0));
  }
  const int end_val = graph->extract_scalar<int64_t>(extra_args.at(1));
  if (!graph->val_is_none(extra_args.at(2))) {
    step_val = graph->extract_scalar<int64_t>(extra_args.at(2));
  }

  const std::vector<int64_t> out_sizes = {
      utils::div_up(end_val - start_val, step_val)};

  graph->virtual_resize(out, out_sizes);
}

void check_arange_input(
    ComputeGraph& graph,
    const ValueRef start,
    const ValueRef end,
    const ValueRef step) {
  if (!graph.val_is_none(start) && !graph.val_is_int(end)) {
    VK_THROW("arange: start must be int!");
  }
  if (!graph.val_is_none(end) && !graph.val_is_int(end)) {
    VK_THROW("arange: end must be int!");
  }
  if (!graph.val_is_none(step) && !graph.val_is_int(end)) {
    VK_THROW("arange: step must be int!");
  }
}

void add_arange_node(
    ComputeGraph& graph,
    const ValueRef start,
    const ValueRef end,
    const ValueRef step,
    const ValueRef out) {
  float start_val = 0.0f;
  float step_val = 1.0f;

  if (graph.val_is_none(end)) {
    VK_THROW("arange: end must be specified!");
  }

  if (!graph.val_is_none(start)) {
    if (graph.val_is_int(start)) {
      start_val = static_cast<float>(graph.extract_scalar<int64_t>(start));
    } else {
      start_val = graph.extract_scalar<float>(start);
    }
  }
  if (!graph.val_is_none(step)) {
    if (graph.val_is_int(step)) {
      step_val = static_cast<float>(graph.extract_scalar<int64_t>(step));
    } else {
      step_val = graph.extract_scalar<float>(step);
    }
  }

  std::string kernel_name("arange");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}},
      // Shader params buffers
      {graph.sizes_ubo(out),
       graph.create_params_buffer(start_val),
       graph.create_params_buffer(step_val)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {start, end, step},
      // Resizing Logic
      resize_arange_node));
}

void arange(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_arange_node(graph, args[0], args[1], args[2], args[7]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.arange.start_step, arange);
}

} // namespace vkcompute
