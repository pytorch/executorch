/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_pool2d_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(in) == WHCN::kChannelsDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kChannelsDim);
}

void resize_pool2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  bool is_max_pool2d = extra_args.at(3) != kDummyValueRef;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  size_t ndim = self_sizes.size();
  std::vector<int64_t> new_out_sizes(ndim);

  // Batch, Channel
  if (ndim == 4) {
    new_out_sizes.at(ndim - 4) = self_sizes.at(ndim - 4);
  }
  new_out_sizes.at(ndim - 3) = self_sizes.at(ndim - 3);

  // Height, Width
  const auto& new_out_sizes_hw = calc_out_sizes_hw(
      *graph,
      self_sizes,
      extra_args.at(0),
      /*kernel_size_only = */ true,
      {extra_args.at(1), extra_args.at(2), extra_args.at(3), extra_args.at(4)});
  new_out_sizes.at(ndim - 2) = new_out_sizes_hw.at(0);
  new_out_sizes.at(ndim - 1) = new_out_sizes_hw.at(1);

  graph->virtual_resize(out, new_out_sizes);

  if (is_max_pool2d) {
    const ValueRef indices = args.at(0).refs.at(1);
    // For max_pool2d variant, indices tensor will be a 0-dim tensor - only
    // resize the indices tensor if this is not the case.
    if (graph->sizes_of(indices).size() > 0) {
      graph->virtual_resize(indices, new_out_sizes);
    }
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
  ValueRef out_tensor = out;
  // Placeholder tensor to fill binding slot for indices tensor in case we are
  // computing max_pool2d instead of max_pool2d_with_indices.
  TmpTensor tmp_indices_tensor =
      TmpTensor(&graph, {}, graph.dtype_of(in), graph.storage_type_of(in));
  ValueRef indices_tensor = tmp_indices_tensor.vref;
  int32_t write_indices = 0;
  if (graph.val_is_value_list(out)) {
    const auto out_val = graph.get_value_list(out);
    out_tensor = out_val->at(0);
    indices_tensor = out_val->at(1);
    write_indices = 1;
  }

  check_pool2d_args(graph, in, out_tensor);

  std::string kernel_name("max_pool2d");
  add_dtype_suffix(kernel_name, graph.dtype_of(out_tensor));

  Kernel2dParams kernel_params = create_kernel2d_params(
      graph,
      kernel_size,
      /*kernel_size_only = */ true,
      stride,
      padding,
      dilation);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{{out_tensor, indices_tensor}, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {
          graph.logical_limits_ubo(out_tensor),
          graph.sizes_ubo(in),
          graph.create_params_buffer(kernel_params),
      },
      // Push Constants
      {},
      // Specialization Constants
      {write_indices},
      // Resize Args
      {kernel_size, stride, padding, dilation, ceil_mode},
      // Resizing Logic
      resize_pool2d_node));
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
  int32_t count_include_pad;
};

DivisorParams create_divisor_params(
    ComputeGraph& graph,
    const ValueRef divisor_override,
    const ValueRef count_include_pad) {
  return {
      graph.val_is_int(divisor_override)
          ? static_cast<int32_t>(graph.get_int(divisor_override))
          : 0,
      int32_t(graph.get_bool(count_include_pad))};
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
  check_pool2d_args(graph, in, out);

  std::string kernel_name("avg_pool2d");
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  Kernel2dParams kernel_params =
      create_kernel2d_params(graph, kernel_size, stride, padding);

  DivisorParams divisor_params =
      create_divisor_params(graph, divisor_override, count_include_pad);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {graph.logical_limits_ubo(out),
       graph.sizes_ubo(in),
       graph.create_params_buffer(kernel_params),
       graph.create_params_buffer(divisor_params)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {kernel_size,
       stride,
       padding,
       /*dilation= */ kDummyValueRef,
       ceil_mode},
      // Resizing Logic
      resize_pool2d_node));
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
  VK_REGISTER_OP(aten.max_pool2d.default, max_pool2d);
}

} // namespace vkcompute
