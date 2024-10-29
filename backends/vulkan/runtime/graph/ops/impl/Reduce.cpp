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

void resize_reduce_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);

  int dim = extra_args[0];

  std::vector<int64_t> new_sizes = in->sizes();
  new_sizes[normalize(dim, new_sizes.size())] = 1;
  out->virtual_resize(new_sizes);
}

void add_reduce_node(
    ComputeGraph& graph,
    ValueRef in,
    const int dim,
    ValueRef out,
    const std::string& op_name) {
  VK_CHECK_COND(
      !graph.is_buffer_storage(in) && !graph.is_buffer_storage(out),
      "Vulkan reduction only supports texture storage");

  const int64_t ndim = graph.dim_of(in);

  int32_t reduce_dim = dim;
  reduce_dim = normalize(reduce_dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  // Check that the concat dim is not the reduction dim, if the tensor has a
  // batch dim greater than 1.
  if (graph.dim_of(in) == 4 && graph.size_at<int>(0, in) > 1) {
    VK_CHECK_COND(graph.concat_dim_of(in) != reduce_dim);
    VK_CHECK_COND(graph.concat_dim_of(out) != reduce_dim);
  }

  vkapi::ShaderInfo shader_descriptor;
  std::string kernel_name = op_name;
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // This should match the value of MAX_NTHREADS in the softmax shader.
  constexpr uint32_t max_nthreads = 16;

  const uint32_t nworkers_per_group = 4;
  const uint32_t ngroups = 4;
  VK_CHECK_COND(nworkers_per_group * ngroups <= max_nthreads);

  utils::uvec3 global_wg_size = graph.logical_limits_of(out);
  global_wg_size[reduce_dim] = 1;

  utils::uvec3 local_wg_size{1, 1, 1};
  local_wg_size[reduce_dim] = nworkers_per_group;
  const int other_dim_1 = (reduce_dim + 1) % 3;
  const int other_dim_2 = (reduce_dim + 2) % 3;
  int32_t group_dim;
  if (global_wg_size[other_dim_1] > global_wg_size[other_dim_2]) {
    local_wg_size[other_dim_1] = ngroups;
    group_dim = other_dim_1;
  } else {
    local_wg_size[other_dim_2] = ngroups;
    group_dim = other_dim_2;
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      // shader_descriptor,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {graph.logical_limits_ubo(in), graph.sizes_ubo(in)},
      // Specialization Constants
      {graph.packed_dim_of(out), reduce_dim, group_dim},
      // Resizing Logic
      resize_reduce_node,
      {dim}));
}

#define DEFINE_REDUCE_FN(op_name, out_arg_idx)                           \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    const IntListPtr dims_list = graph.get_int_list(args[1]);            \
    VK_CHECK_COND(dims_list->size() == 1);                               \
    return add_reduce_node(                                              \
        graph, args[0], dims_list->at(0), args[out_arg_idx], #op_name);  \
  }

DEFINE_REDUCE_FN(sum, 4)
DEFINE_REDUCE_FN(mean, 4)
DEFINE_REDUCE_FN(amax, 3)
DEFINE_REDUCE_FN(amin, 3)

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.sum.dim_IntList, sum);
  VK_REGISTER_OP(aten.mean.dim, mean);
  VK_REGISTER_OP(aten.amax.default, amax);
  VK_REGISTER_OP(aten.amin.default, amin);
}

} // namespace vkcompute
