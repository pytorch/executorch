/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using namespace utils;

// Custom global workgroup size function for var_buffer
GlobalWorkGrid var_buffer_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const int dim = resize_args.at(0);

  const int64_t ndim = graph->dim_of(in);
  int32_t reduce_dim = normalize(dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  utils::uvec3 lwg_extents{1u, 1u, 1u};
  lwg_extents[reduce_dim] = 4u;
  return GlobalWorkGrid(
      {graph->size_at<uint32_t>(-1, out),
       graph->size_at<uint32_t>(-2, out),
       graph->size_at<uint32_t>(-3, out) * graph->size_at<uint32_t>(-4, out)},
      kTiledWorkGrid,
      LocalWorkGroup(lwg_extents));
}

// Custom global workgroup size function for var_texture
GlobalWorkGrid var_texture_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const int dim = resize_args.at(0);

  const int64_t ndim = graph->dim_of(in);
  int32_t reduce_dim = normalize(dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  utils::uvec3 extents = graph->logical_limits_of(out);
  extents[reduce_dim] = 1;
  utils::uvec3 lwg_extents{1u, 1u, 1u};
  lwg_extents[reduce_dim] = 4u;
  const int other_dim_1 = (reduce_dim + 1) % 3;
  const int other_dim_2 = (reduce_dim + 2) % 3;
  if (extents[other_dim_1] > extents[other_dim_2]) {
    lwg_extents[other_dim_1] = 4u;
  } else {
    lwg_extents[other_dim_2] = 4u;
  }
  return GlobalWorkGrid(extents, kTiledWorkGrid, LocalWorkGroup(lwg_extents));
}

void resize_var_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  const int dim = extra_args.at(0);

  std::vector<int64_t> new_sizes = graph->sizes_of(in);
  if (!new_sizes.empty()) {
    new_sizes.at(normalize(dim, new_sizes.size())) = 1;
  }

  graph->virtual_resize(out, new_sizes);
}

void add_var_buffer_node(
    ComputeGraph& graph,
    ValueRef in,
    const int dim,
    bool unbiased,
    ValueRef out) {
  const int64_t ndim = graph.dim_of(in);
  int32_t reduce_dim = normalize(dim, ndim);
  reduce_dim = nchw_dim_to_whcn_dim(reduce_dim, ndim);

  // Check that the concat dim is not the reduction dim, if the tensor has a
  // batch dim greater than 1
  if (graph.dim_of(in) == 4 && graph.size_at<int>(0, in) > 1) {
    VK_CHECK_COND(graph.concat_dim_of(in) != reduce_dim);
    VK_CHECK_COND(graph.concat_dim_of(out) != reduce_dim);
  }

  std::string kernel_name = "var";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  std::vector<PushConstantDataInfo> push_constants;
  int32_t unbiased_int = static_cast<int32_t>(unbiased);
  push_constants.emplace_back(&unbiased_int, sizeof(unbiased_int));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      var_buffer_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {
          graph.sizes_ubo(in),
          graph.strides_ubo(in),
          graph.sizes_ubo(out),
          graph.strides_ubo(out),
      },
      // Push Constants
      push_constants,
      // Specialization Constants
      {reduce_dim},
      // Resize Args
      {dim},
      // Resizing Logic
      resize_var_node));
}

void add_var_texture_node(
    ComputeGraph& graph,
    ValueRef in,
    const int dim,
    bool unbiased,
    ValueRef out) {
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

  std::string kernel_name = "var";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  // This should match the value of MAX_NTHREADS in the softmax shader.
  constexpr uint32_t max_nthreads = 16;

  const uint32_t nworkers_per_group = 4;
  const uint32_t ngroups = 4;
  VK_CHECK_COND(nworkers_per_group * ngroups <= max_nthreads);

  utils::uvec3 extents = graph.logical_limits_of(out);
  extents[reduce_dim] = 1;
  const int other_dim_1 = (reduce_dim + 1) % 3;
  const int other_dim_2 = (reduce_dim + 2) % 3;
  int32_t group_dim;
  if (extents[other_dim_1] > extents[other_dim_2]) {
    group_dim = other_dim_1;
  } else {
    group_dim = other_dim_2;
  }

  std::vector<PushConstantDataInfo> push_constants;
  int32_t unbiased_int = static_cast<int32_t>(unbiased);
  push_constants.emplace_back(&unbiased_int, sizeof(unbiased_int));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      var_texture_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      {graph.logical_limits_ubo(in), graph.sizes_ubo(in)},
      // Push Constants
      push_constants,
      // Specialization Constants
      {graph.packed_dim_of(out), reduce_dim, group_dim},
      // Resize Args
      {dim},
      // Resizing Logic
      resize_var_node));
}

void add_var_node(
    ComputeGraph& graph,
    ValueRef in,
    const int dim,
    bool unbiased,
    ValueRef out) {
  bool is_buffer = graph.is_buffer_storage(in) || graph.is_buffer_storage(out);

  if (is_buffer) {
    add_var_buffer_node(graph, in, dim, unbiased, out);
  } else {
    add_var_texture_node(graph, in, dim, unbiased, out);
  }
}

void var(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const IntListPtr dims_list = graph.get_int_list(args[1]);
  VK_CHECK_COND(dims_list->size() == 1);
  bool unbiased = true;
  if (args.size() > 2) {
    unbiased = graph.get_bool(args[2]);
  }
  return add_var_node(
      graph, args[0], static_cast<int>(dims_list->at(0)), unbiased, args[4]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.var.dim, var);
}

} // namespace vkcompute
