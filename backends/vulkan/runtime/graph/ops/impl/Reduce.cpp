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

void resize_reduce_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  const int32_t reduce_dim_nchw =
      graph->extract_scalar<int32_t>(resize_args.at(0));

  std::vector<int64_t> new_sizes = graph->sizes_of(in);
  new_sizes.at(normalize(reduce_dim_nchw, new_sizes.size())) = 1;
  graph->virtual_resize(out, new_sizes);
}

void resize_reduce2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  // Extract the dimensions to reduce over
  const std::vector<int64_t> dims_list =
      graph->extract_int_or_symint_list(resize_args.at(0));
  int32_t reduce_dim1_nchw = dims_list[0];
  int32_t reduce_dim2_nchw = dims_list[1];

  std::vector<int64_t> new_sizes = graph->sizes_of(in);
  new_sizes.at(normalize(reduce_dim1_nchw, new_sizes.size())) = 1;
  new_sizes.at(normalize(reduce_dim2_nchw, new_sizes.size())) = 1;
  graph->virtual_resize(out, new_sizes);
}

utils::uvec3 reduce_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef out = args.at(0).refs.at(0);
  const int32_t reduce_dim_whcn =
      graph->extract_scalar<int32_t>(resize_args.at(1));

  utils::uvec3 global_wg_size = graph->logical_limits_of(out);
  global_wg_size[reduce_dim_whcn] = 1;
  return global_wg_size;
}

utils::uvec3 reduce_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)global_workgroup_size;

  const int32_t reduce_dim_whcn =
      graph->extract_scalar<int32_t>(resize_args.at(1));
  const int64_t group_dim_whcn =
      graph->extract_scalar<int64_t>(resize_args.at(2));

  // This should match the value of MAX_NTHREADS in the reduce shader.
  constexpr uint32_t max_nthreads = 16;

  const uint32_t nworkers_per_group = 4;
  const uint32_t ngroups = 4;
  VK_CHECK_COND(nworkers_per_group * ngroups <= max_nthreads);

  utils::uvec3 local_wg_size{1, 1, 1};
  local_wg_size[reduce_dim_whcn] = nworkers_per_group;
  local_wg_size[group_dim_whcn] = ngroups;

  return local_wg_size;
}

void add_reduce_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef out,
    const std::string& op_name) {
  VK_CHECK_COND(
      !graph.is_buffer_storage(in) && !graph.is_buffer_storage(out),
      "Vulkan reduction only supports texture storage");

  const int64_t ndim = graph.dim_of(in);

  int32_t reduce_dim = graph.extract_scalar<int32_t>(dim_ref);
  reduce_dim = normalize(reduce_dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  // Check that the concat dim is not the reduction dim, if the tensor has a
  // batch dim greater than 1.
  if (graph.dim_of(in) == 4 && graph.size_at<int>(0, in) > 1) {
    VK_CHECK_COND(graph.concat_dim_of(in) != reduce_dim);
    VK_CHECK_COND(graph.concat_dim_of(out) != reduce_dim);
  }

  std::string kernel_name = op_name;
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // Calculate group_dim for specialization constants
  const int other_dim_1 = (reduce_dim + 1) % 3;
  const int other_dim_2 = (reduce_dim + 2) % 3;
  int32_t group_dim;
  utils::uvec3 limits = graph.logical_limits_of(out);
  if (limits[other_dim_1] > limits[other_dim_2]) {
    group_dim = other_dim_1;
  } else {
    group_dim = other_dim_2;
  }

  const ValueRef reduce_dim_whcn_ref =
      graph.get_or_add_value_for_int(reduce_dim);
  const ValueRef group_dim_whcn_ref = graph.get_or_add_value_for_int(group_dim);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      reduce_global_wg_size,
      reduce_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {graph.logical_limits_ubo(in), graph.sizes_ubo(in)},
      // Push Constants
      {},
      // Specialization Constants
      {graph.packed_dim_of(out), reduce_dim, group_dim},
      // Resize Args
      {dim_ref, reduce_dim_whcn_ref, group_dim_whcn_ref},
      // Resizing Logic
      resize_reduce_node));
}

void add_reduce2d_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dims_ref,
    const ValueRef out,
    const std::string& op_name) {
  VK_CHECK_COND(
      !graph.is_buffer_storage(in) && !graph.is_buffer_storage(out),
      "Vulkan reduction only supports texture storage");

  const int64_t ndim = graph.dim_of(in);

  // Extract the two dimensions to reduce over
  const std::vector<int64_t> dims_list =
      graph.extract_int_or_symint_list(dims_ref);
  VK_CHECK_COND(
      dims_list.size() == 2, "reduce2d requires exactly 2 dimensions");

  int32_t reduce_dim1 = normalize(dims_list[0], ndim);
  int32_t reduce_dim2 = normalize(dims_list[1], ndim);

  // Convert to WHCN format
  reduce_dim1 = nchw_dim_to_whcn_dim(reduce_dim1, ndim);
  reduce_dim2 = nchw_dim_to_whcn_dim(reduce_dim2, ndim);

  // Check that none of the reduction dims are packed
  VK_CHECK_COND(graph.packed_dim_of(in) != reduce_dim1);
  VK_CHECK_COND(graph.packed_dim_of(in) != reduce_dim2);
  VK_CHECK_COND(graph.packed_dim_of(out) != reduce_dim1);
  VK_CHECK_COND(graph.packed_dim_of(out) != reduce_dim2);

  // Check that the concat dim is not one of the reduction dims
  if (graph.dim_of(in) == 4 && graph.size_at<int>(0, in) > 1) {
    VK_CHECK_COND(graph.concat_dim_of(in) != reduce_dim1);
    VK_CHECK_COND(graph.concat_dim_of(in) != reduce_dim2);
    VK_CHECK_COND(graph.concat_dim_of(out) != reduce_dim1);
    VK_CHECK_COND(graph.concat_dim_of(out) != reduce_dim2);
  }

  std::string kernel_name = op_name + "2d"; // Add "2d" suffix
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // Calculate group_dim for specialization constants (use remaining dimension)
  int32_t group_dim = 0;
  for (int i = 0; i < 3; i++) {
    if (i != reduce_dim1 && i != reduce_dim2) {
      group_dim = i;
      break;
    }
  }

  const ValueRef reduce_dim1_whcn_ref =
      graph.get_or_add_value_for_int(reduce_dim1);
  const ValueRef reduce_dim2_whcn_ref =
      graph.get_or_add_value_for_int(reduce_dim2);
  const ValueRef group_dim_whcn_ref = graph.get_or_add_value_for_int(group_dim);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      reduce_global_wg_size,
      reduce_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {graph.logical_limits_ubo(in), graph.sizes_ubo(in)},
      // Push Constants
      {},
      // Specialization Constants
      {graph.packed_dim_of(out), reduce_dim1, reduce_dim2, group_dim},
      // Resize Args
      {dims_ref,
       reduce_dim1_whcn_ref,
       reduce_dim2_whcn_ref,
       group_dim_whcn_ref},
      // Resizing Logic
      resize_reduce2d_node));
}

#define DEFINE_REDUCE_FN(op_name, out_arg_idx)                           \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    const std::vector<int64_t> dims_list =                               \
        graph.extract_int_or_symint_list(args[1]);                       \
    if (dims_list.size() == 1) {                                         \
      const int64_t dim_val = dims_list.at(0);                           \
      const ValueRef dim_ref = graph.get_or_add_value_for_int(dim_val);  \
      return add_reduce_node(                                            \
          graph, args[0], dim_ref, args[out_arg_idx], #op_name);         \
    }                                                                    \
    if (dims_list.size() == 2) {                                         \
      return add_reduce2d_node(                                          \
          graph, args[0], args[1], args[out_arg_idx], #op_name);         \
    }                                                                    \
    VK_CHECK_COND(false, "Only 1 or 2 dimensions supported");            \
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
