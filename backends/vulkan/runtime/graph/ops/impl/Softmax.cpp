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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using namespace utils;

utils::uvec3 pick_softmax_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const int32_t reduce_dim_xyz =
      graph->extract_scalar<int32_t>(resize_args.at(1));

  utils::uvec3 global_size = graph->logical_limits_of(out);
  global_size[reduce_dim_xyz] = 1;
  return global_size;
}

utils::uvec3 pick_softmax_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)global_workgroup_size;
  (void)args;

  const int64_t group_dim_xyz =
      graph->extract_scalar<int64_t>(resize_args.at(2));

  const int32_t reduce_dim_xyz =
      graph->extract_scalar<int32_t>(resize_args.at(1));

  // These values are hardcoded in add_softmax_node
  const uint32_t nworkers_per_group = 4;
  const uint32_t ngroups = 4;

  utils::uvec3 local_wg_size{1, 1, 1};
  local_wg_size[reduce_dim_xyz] = nworkers_per_group;
  local_wg_size[group_dim_xyz] = ngroups;

  return local_wg_size;
}

void resize_softmax_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  graph->virtual_resize(out, in_sizes);
}

void add_softmax_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef out,
    bool log_softmax) {
  VK_CHECK_COND(
      !graph.is_buffer_storage(in) && !graph.is_buffer_storage(out),
      "Vulkan softmax only supports texture storage");

  const int64_t ndim = graph.dim_of(in);

  int32_t reduce_dim_nchw = graph.extract_scalar<int32_t>(dim_ref);
  reduce_dim_nchw = normalize(reduce_dim_nchw, ndim);
  const int32_t reduce_dim_xyz = nchw_dim_to_whcn_dim(reduce_dim_nchw, ndim);

  // Check that the concat dim is not the reduction dim, if the tensor has a
  // batch dim greater than 1.
  if (graph.dim_of(in) == 4 && graph.size_at<int>(0, in) > 1) {
    VK_CHECK_COND(
        graph.concat_dim_of(in) != reduce_dim_xyz,
        "Softmax shader currently does not support concat dim == reduce dim");
    VK_CHECK_COND(
        graph.concat_dim_of(out) != reduce_dim_xyz,
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

  // Determine the group dimension
  const int other_dim_1 = (reduce_dim_xyz + 1) % 3;
  const int other_dim_2 = (reduce_dim_xyz + 2) % 3;
  int32_t group_dim;
  utils::uvec3 global_wg_size = graph.logical_limits_of(out);
  if (global_wg_size[other_dim_1] > global_wg_size[other_dim_2]) {
    group_dim = other_dim_1;
  } else {
    group_dim = other_dim_2;
  }

  const ValueRef reduce_dim_xyz_ref =
      graph.get_or_add_value_for_int(reduce_dim_xyz);
  const ValueRef group_dim_xyz_ref = graph.get_or_add_value_for_int(group_dim);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_softmax_global_wg_size,
      pick_softmax_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(in), graph.logical_limits_pc_of(out)},
      // Specialization Constants
      {graph.packed_dim_of(out), reduce_dim_xyz, group_dim},
      // Resize Args
      {dim_ref, reduce_dim_xyz_ref, group_dim_xyz_ref},
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
