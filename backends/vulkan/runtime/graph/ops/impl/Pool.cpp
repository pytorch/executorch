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

namespace vkcompute {

void check_pool2d_args(const api::vTensor& in, const api::vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(in, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, utils::kChannelsPacked));
}

void resize_pool2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  bool is_max_pool2d = extra_args[3] != kDummyValueRef;

  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr self = graph->get_tensor(args[1].refs[0]);

  size_t ndim = self->sizes().size();
  std::vector<int64_t> new_out_sizes(ndim);

  // Batch, Channel
  if (ndim == 4) {
    new_out_sizes.at(ndim - 4) = self->sizes().at(ndim - 4);
  }
  new_out_sizes.at(ndim - 3) = self->sizes().at(ndim - 3);

  // Height, Width
  const auto& new_out_sizes_hw = calc_out_sizes_hw(
      *graph,
      self->sizes(),
      extra_args[0],
      /*kernel_size_only = */ true,
      {extra_args[1], extra_args[2], extra_args[3], extra_args[4]});
  new_out_sizes.at(ndim - 2) = new_out_sizes_hw.at(0);
  new_out_sizes.at(ndim - 1) = new_out_sizes_hw.at(1);

  out->virtual_resize(new_out_sizes);

  if (is_max_pool2d) {
    vTensorPtr indices = graph->get_tensor(args[0].refs[1]);
    indices->virtual_resize(new_out_sizes);
  }
}

//
// max_pool2d
//

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
  vTensorPtr t_in = graph.get_tensor(arg);

  const auto out_val = graph.get_value_list(out);
  vTensorPtr t_out = graph.get_tensor(out_val->at(0));

  check_pool2d_args(*t_in, *t_out);

  utils::uvec3 global_size = t_out->image_extents();
  utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::string kernel_name("max_pool2d");
  add_dtype_suffix(kernel_name, *t_out);

  Kernel2dParams kernel_params = create_kernel2d_params(
      graph,
      kernel_size,
      /*kernel_size_only = */ true,
      stride,
      padding,
      dilation);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{{out_val->at(0), out_val->at(1)}, vkapi::MemoryAccessType::WRITE},
       {arg, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {
          t_out->texture_limits_ubo(),
          t_in->sizes_ubo(),
          graph.create_params_buffer(kernel_params),
      },
      // Specialization Constants
      {},
      // Resizing Logic
      resize_pool2d_node,
      {kernel_size, stride, padding, dilation, ceil_mode}));
}

void max_pool2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_max_pool2d_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
}

//
// avg_pool2d
//

struct DivisorParams final {
  int32_t divisor_override;
  bool count_include_pad;
};

DivisorParams create_divisor_params(
    ComputeGraph& graph,
    const ValueRef divisor_override,
    const ValueRef count_include_pad) {
  return {
      graph.val_is_int(divisor_override)
          ? static_cast<int32_t>(graph.get_int(divisor_override))
          : 0,
      graph.get_bool(count_include_pad)};
}

void add_avg_pool2d_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef ceil_mode,
    const ValueRef count_include_pad,
    const ValueRef divisor_override,
    const ValueRef out) {
  ValueRef arg = prepack_if_tensor_ref(graph, in);
  vTensorPtr t_in = graph.get_tensor(arg);
  vTensorPtr t_out = graph.get_tensor(out);

  check_pool2d_args(*t_in, *t_out);

  utils::uvec3 global_size = t_out->image_extents();
  utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::string kernel_name("avg_pool2d");
  add_dtype_suffix(kernel_name, *t_out);

  Kernel2dParams kernel_params =
      create_kernel2d_params(graph, kernel_size, stride, padding);

  DivisorParams divisor_params =
      create_divisor_params(graph, divisor_override, count_include_pad);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {arg, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->texture_limits_ubo(),
       t_in->sizes_ubo(),
       graph.create_params_buffer(kernel_params),
       graph.create_params_buffer(divisor_params)},
      // Specialization Constants
      {},
      // Resizing Logic
      resize_pool2d_node,
      {kernel_size,
       stride,
       padding,
       /*dilation= */ kDummyValueRef,
       ceil_mode}));
}

void avg_pool2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_avg_pool2d_node(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args[6],
      args[7]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.avg_pool2d.default, avg_pool2d);
  VK_REGISTER_OP(aten.max_pool2d_with_indices.default, max_pool2d);
}

} // namespace vkcompute
