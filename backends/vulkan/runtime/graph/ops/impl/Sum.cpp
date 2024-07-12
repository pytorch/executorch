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
#include <cstdint>
#include <set>

namespace vkcompute {

std::vector<int64_t>
calc_out_sizes(api::vTensor& self, int64_t dim, bool keepdim) {
  std::vector<int64_t> output_size = self.sizes();
  if (keepdim) {
    output_size.at(dim) = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }
  return output_size;
}

void resize_sum_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;
  vTensorPtr out = graph->get_tensor(extra_args[0]);
  vTensorPtr in = graph->get_tensor(extra_args[1]);

  const auto dim = extra_args[2];
  const auto keepdim = extra_args[3];

  std::vector<int64_t> output_size = calc_out_sizes(*in, dim, keepdim);

  out->virtual_resize(output_size);
}

void check_sum_args(const api::vTensor& in, const api::vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(in, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, utils::kChannelsPacked));
}

void add_sum_dim_node(
    ComputeGraph& graph,
    const ValueRef in,
    const int64_t dim,
    const bool keepdim,
    const ValueRef out) {
  ValueRef arg = prepack_if_tensor_ref(graph, in);

  vTensorPtr t_out = graph.get_tensor(out);
  vTensorPtr t_input = graph.get_tensor(in);

  check_sum_args(*t_input, *t_out);

  int64_t in_dim = t_input->sizes().size();
  int32_t channel =
      in_dim > 2 ? static_cast<int32_t>(t_input->sizes()[in_dim - 3]) : 1;
  uint32_t dim_size = t_input->sizes()[dim];

  std::string kernel_name("sum_dim");
  kernel_name.reserve(kShaderNameReserve);
  if (keepdim) {
    kernel_name += "_keepdim";
  }
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {arg, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->texture_limits_ubo(),
       graph.create_params_buffer(dim + 4 - in_dim),
       graph.create_params_buffer(dim_size),
       graph.create_params_buffer(int(ceil(channel / 4.0)))},
      // Specialization Constants
      {},
      // Resizing Logic
      resize_sum_node,
      {out, in, static_cast<int>(dim), keepdim}));
}

ValueRef add_node(
    ComputeGraph& graph,
    const ValueRef input,
    const int dim,
    const bool keepdim,
    const vkapi::ScalarType dtype = vkapi::kFloat) {
  std::vector<int64_t> output_size =
      calc_out_sizes(*(graph.get_tensor(input)), dim, keepdim);
  return graph.add_tensor(output_size, dtype, utils::kChannelsPacked);
}

void add_sum_dim_IntList(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef opt_dim,
    const ValueRef keepdim,
    const ValueRef out) {
  bool keepdim_val = graph.get_bool(keepdim);

  std::set<int64_t> dims_set;
  const auto dims_to_sum = *graph.get_int_list(opt_dim);
  int64_t in_dim = graph.get_tensor(in)->sizes().size();

  if (dims_to_sum.empty()) {
    // If dim is not specified, reduce over all dims
    for (int64_t i = 0; i < in_dim; ++i) {
      dims_set.insert(i);
    }
  } else {
    for (const auto& dim : dims_to_sum) {
      // Normalize (negative) dim into range [0, self.dim() - 1]
      int64_t dim_normalized = normalize(dim, in_dim);
      dims_set.insert(dim_normalized);
    }
  }

  // Reduce the higher dimensionalities first, otherwise when keepdim is
  // false, it will be reducing the wrong dimension.
  // We add intermediate nodes before the final output node, so we traverse
  // until `std::prev(dims_set.rend())`. The final output node is added after
  // the for loop.
  ValueRef input = in;
  for (auto dim = dims_set.rbegin(); dim != std::prev(dims_set.rend()); ++dim) {
    ValueRef tmp_node = add_node(graph, input, *dim, keepdim_val);
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
