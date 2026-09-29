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

#include <cmath>

namespace vkcompute {

void resize_arange_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);

  double start_val = 0.0;
  double step_val = 1.0;
  if (!graph->val_is_none(extra_args.at(0))) {
    start_val = graph->extract_scalar<double>(extra_args.at(0));
  }
  const double end_val = graph->extract_scalar<double>(extra_args.at(1));
  if (!graph->val_is_none(extra_args.at(2))) {
    step_val = graph->extract_scalar<double>(extra_args.at(2));
  }

  VK_CHECK_COND(step_val != 0.0, "arange: step must be nonzero");
  const double range_size = (end_val - start_val) / step_val;
  VK_CHECK_COND(
      range_size >= 0.0, "arange: bounds are inconsistent with step sign");
  const std::vector<int64_t> out_sizes = {
      static_cast<int64_t>(std::ceil(range_size))};

  graph->virtual_resize(out, out_sizes);
}

void check_arange_input(
    ComputeGraph& graph,
    const ValueRef start,
    const ValueRef end,
    const ValueRef step) {
  if (!graph.val_is_none(start) && !graph.val_is_int(start)) {
    VK_THROW("arange: start must be int!");
  }
  if (!graph.val_is_none(end) && !graph.val_is_int(end)) {
    VK_THROW("arange: end must be int!");
  }
  if (!graph.val_is_none(step) && !graph.val_is_int(step)) {
    VK_THROW("arange: step must be int!");
  }
}

vkapi::BufferBindInfo get_arange_param_buffer(
    ComputeGraph& graph,
    const ValueRef value,
    const float default_value) {
  if (graph.val_is_symint(value)) {
    return graph.get_or_create_int_param_buffer(value);
  }
  return graph.create_params_buffer(
      graph.extract_scalar_or<float>(value, default_value));
}

void add_arange_node(
    ComputeGraph& graph,
    const ValueRef start,
    const ValueRef end,
    const ValueRef step,
    const ValueRef out) {
  if (graph.val_is_none(end)) {
    VK_THROW("arange: end must be specified!");
  }

  std::string kernel_name("arange");
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  const utils::ivec2 params_are_int = {
      graph.val_is_symint(start) ? 1 : 0, graph.val_is_symint(step) ? 1 : 0};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}},
      // Shader params buffers
      {graph.meta_ubo(out),
       get_arange_param_buffer(graph, start, 0.0f),
       get_arange_param_buffer(graph, step, 1.0f)},
      // Push Constants
      {PushConstantDataInfo(&params_are_int, sizeof(params_are_int))},
      // Specialization Constants
      {graph.hashed_layout_of(out)},
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
