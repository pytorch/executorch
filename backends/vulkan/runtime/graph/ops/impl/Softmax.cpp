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
  VK_CHECK_COND(
      !graph.is_buffer_storage(in) && !graph.is_buffer_storage(out),
      "Vulkan softmax only supports texture storage");

  const int64_t ndim = graph.dim_of(in);

  int32_t reduce_dim = graph.extract_scalar<int32_t>(dim);
  reduce_dim = normalize(reduce_dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  // Check that the concat dim is not the reduction dim, if the tensor has a
  // batch dim greater than 1.
  if (graph.dim_of(in) == 4 && graph.size_at<int>(0, in) > 1) {
    VK_CHECK_COND(
        graph.concat_dim_of(in) != reduce_dim,
        "Softmax shader currently does not support concat dim == reduce dim");
    VK_CHECK_COND(
        graph.concat_dim_of(out) != reduce_dim,
        "Softmax shader currently does not support concat dim == reduce dim");
  }

  vkapi::ShaderInfo shader_descriptor;
  std::string kernel_name = "softmax";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  if (log_softmax) {
    kernel_name = "log_" + kernel_name;
  }

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
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {graph.logical_limits_ubo(out), graph.sizes_ubo(in)},
      // Specialization Constants
      {graph.packed_dim_of(out), reduce_dim, group_dim},
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
