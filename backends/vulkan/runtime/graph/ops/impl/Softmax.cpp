/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using namespace utils;

void resize_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);

  std::vector<int64_t> in_sizes = in->sizes();
  out->virtual_resize(in_sizes);
}

void add_softmax_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef dim,
    ValueRef out,
    bool log_softmax) {
  ValueRef in_arg = prepack_if_tensor_ref(graph, in);
  vTensorPtr t_in = graph.get_tensor(in_arg);
  int64_t in_dim = t_in->dim();

  int64_t softmax_dim = graph.extract_scalar<int64_t>(dim);
  softmax_dim = normalize(softmax_dim, in_dim);

  vTensorPtr t_out = graph.get_tensor(out);

  vkapi::ShaderInfo shader_descriptor;
  std::string kernel_name = in_dim - softmax_dim == 3
      ? "softmax_channel"
      : "softmax_batch_height_width";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);
  if (log_softmax) {
    kernel_name = "log_" + kernel_name;
  }

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      // shader_descriptor,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {in_arg, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->texture_limits_ubo(),
       t_in->sizes_ubo(),
       graph.create_params_buffer(utils::make_ivec2({in_dim, softmax_dim}))},
      // Specialization Constants
      {},
      // Resizing Logic
      resize_softmax_node));
}

void softmax(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[1] bool half_to_float is unused
  return add_softmax_node(
      graph, args[0], args[1], args[3], /* log_softmax = */ false);
}

void log_softmax(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[1] bool half_to_float is unused
  return add_softmax_node(
      graph, args[0], args[1], args[3], /* log_softmax = */ true);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten._softmax.default, softmax);
  VK_REGISTER_OP(aten._log_softmax.default, log_softmax);
}

} // namespace vkcompute
