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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/StatsUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <cstdint>
#include <set>

namespace vkcompute {

void resize_sum_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;
  vTensor& out = graph->get_val(extra_args[0]).toTensor();
  vTensor& in = graph->get_val(extra_args[1]).toTensor();

  const auto dim = extra_args[2];
  const auto keepdim = extra_args[3];

  std::vector<int64_t> output_size = calc_out_sizes(in.sizes(), dim, keepdim);

  out.virtual_resize(output_size);
}

void check_sum_args(const vTensor& in, const vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(in, api::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, api::kChannelsPacked));
}

void add_sum_dim_node(
    ComputeGraph& graph,
    const ValueRef in,
    const int64_t dim,
    const bool keepdim,
    const ValueRef out) {
  ValueRef arg = prepack_if_tensor_ref(graph, in);

  vTensor& t_out = graph.get_val(out).toTensor();
  vTensor& t_input = graph.get_val(in).toTensor();

  check_sum_args(t_input, t_out);

  int64_t in_dim = t_input.sizes().size();
  int32_t channel =
      in_dim > 2 ? static_cast<int32_t>(t_input.sizes()[in_dim - 3]) : 1;
  uint32_t dim_size = t_input.sizes()[dim];

  api::utils::uvec3 global_size = t_out.virtual_extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::string kernel_name("sum_dim");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}, {arg, api::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out.extents_ubo(),
       graph.create_params_buffer(dim + 4 - in_dim),
       graph.create_params_buffer(dim_size),
       graph.create_params_buffer(int(ceil(channel / 4.0))),
       graph.create_params_buffer(keepdim)},
      // Resizing
      resize_sum_node,
      {out, in, static_cast<int>(dim), keepdim}));
}

void add_sum_dim_IntList(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef opt_dim,
    const ValueRef keepdim,
    const ValueRef out) {
  bool keepdim_val = graph.get_val(keepdim).toBool();
  vTensor& in_tensor = graph.get_val(in).toTensor();

  const auto& dims_to_sum = graph.get_val(opt_dim).toIntList();
  int64_t in_dim = in_tensor.sizes().size();
  std::set<int64_t> dims_set = calc_dims_to_aggregate(dims_to_sum, in_dim);

  // Reduce the higher dimensionalities first, otherwise when keepdim is
  // false, it will be reducing the wrong dimension.
  // We add intermediate nodes before the final output node, so we traverse
  // until `std::prev(dims_set.rend())`. The final output node is added after
  // the for loop.
  ValueRef input = in;
  for (auto dim = dims_set.rbegin(); dim != std::prev(dims_set.rend()); ++dim) {
    const auto in_sizes = graph.get_val(input).toTensor().sizes();
    std::vector<int64_t> out_sizes =
        calc_out_sizes(in_sizes, *dim, keepdim_val);

    ValueRef tmp_node = graph.add_tensor(
        out_sizes, api::kFloat, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
    add_sum_dim_node(graph, input, *dim, keepdim_val, tmp_node);
    input = tmp_node;
  }
  // We add the final output node.
  add_sum_dim_node(graph, input, *dims_set.begin(), keepdim_val, out);
}

void sum_dim_IntList(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[3] represents dtype, however serialization of `ScalarType` is not
  // supported yet. Since our usecase for this op is always float/half, it's
  // removed from parameters for now.
  return add_sum_dim_IntList(graph, args[0], args[1], args[2], args[4]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.sum.dim_IntList, sum_dim_IntList);
}

} // namespace vkcompute
