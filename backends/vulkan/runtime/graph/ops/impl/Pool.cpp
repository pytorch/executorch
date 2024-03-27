/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace at {
namespace native {
namespace vulkan {

void resize_max_pool2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensor& out = graph->get_val(args[0].refs[0]).toTensor();
  vTensor& indices = graph->get_val(args[0].refs[1]).toTensor();
  vTensor& self = graph->get_val(args[1].refs[0]).toTensor();

  size_t ndim = self.sizes().size();
  std::vector<int64_t> new_out_sizes(ndim);

  // Batch
  if (ndim == 4) {
    new_out_sizes.at(ndim - 4) = self.sizes().at(ndim - 4);
  }
  // Channel
  new_out_sizes.at(ndim - 3) = self.sizes().at(ndim - 3);

  const auto kernel_size = reverse(*graph, extra_args[0]);
  const auto stride = reverse(*graph, extra_args[1]);
  const auto padding = reverse(*graph, extra_args[2]);
  const auto dilation = reverse(*graph, extra_args[3]);
  const bool ceil_mode = graph->get_val(extra_args[4]).toBool();

  // Height
  new_out_sizes.at(ndim - 2) = calc_out_size(
      self.sizes().at(ndim - 2),
      kernel_size.data[1],
      stride.data[1],
      padding.data[1],
      dilation.data[1],
      ceil_mode);
  // Width
  new_out_sizes.at(ndim - 1) = calc_out_size(
      self.sizes().at(ndim - 1),
      kernel_size.data[0],
      stride.data[0],
      padding.data[0],
      dilation.data[0],
      ceil_mode);

  VK_CHECK_COND(new_out_sizes.at(ndim - 2) >= 1);
  VK_CHECK_COND(new_out_sizes.at(ndim - 1) >= 1);

  out.virtual_resize(new_out_sizes);
  indices.virtual_resize(new_out_sizes);
}

void check_max_pool2d_args(const vTensor& in, const vTensor& out) {
  VK_CHECK_COND(
      check_memory_layout_is(in, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED));
  VK_CHECK_COND(check_memory_layout_is(
      out, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED));
}

void add_max_pool2d_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef ceil_mode,
    const ValueRef out) {
  ValueRef arg = prepack_if_tensor_ref(graph, in);
  vTensor& t_in = graph.get_val(arg).toTensor();

  const auto& out_val = graph.get_val(out).toValueList();
  vTensor& t_out = graph.get_val(out_val[0]).toTensor();

  check_max_pool2d_args(t_in, t_out);

  api::utils::uvec3 global_size = t_out.virtual_extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::stringstream kernel_name;
  kernel_name << "max_pool2d";
  apply_dtype_suffix(kernel_name, t_out);

  KernelParams kernel_params{
      reverse(graph, kernel_size),
      reverse(graph, stride),
      reverse(graph, padding),
      reverse(graph, dilation),
  };

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name.str()),
      global_size,
      local_size,
      // Inputs and Outputs
      {{{out_val[0], out_val[1]}, api::MemoryAccessType::WRITE},
       {arg, api::MemoryAccessType::READ}},
      // Shader params buffers
      {
          t_out.extents_ubo(),
          t_in.extents_ubo(),
          graph.create_params_buffer(kernel_params),
      },
      // Resizing
      resize_max_pool2d_node,
      {kernel_size, stride, padding, dilation, ceil_mode}));
}

void max_pool2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_max_pool2d_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.max_pool2d_with_indices.default, max_pool2d);
}

} // namespace vulkan
} // namespace native
} // namespace at
