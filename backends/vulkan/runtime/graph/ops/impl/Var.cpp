/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using namespace utils;

void resize_var_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);

  int dim = extra_args[0];

  std::vector<int64_t> new_sizes = in->sizes();
  if (!new_sizes.empty()) {
    new_sizes.at(normalize(dim, new_sizes.size())) = 1;
  }
  out->virtual_resize(new_sizes);
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

  const uint32_t nworkers_per_group = 4;

  utils::uvec3 global_wg_size = {
      graph.size_at<uint32_t>(-1, out),
      graph.size_at<uint32_t>(-2, out),
      graph.size_at<uint32_t>(-3, out) * graph.size_at<uint32_t>(-4, out)};

  utils::uvec3 local_wg_size{1, 1, 1};
  local_wg_size[reduce_dim] = nworkers_per_group;

  std::vector<PushConstantDataInfo> push_constants;
  int32_t unbiased_int = static_cast<int32_t>(unbiased);
  push_constants.emplace_back(&unbiased_int, sizeof(unbiased_int));

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      local_wg_size,
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

  std::vector<PushConstantDataInfo> push_constants;
  int32_t unbiased_int = static_cast<int32_t>(unbiased);
  push_constants.emplace_back(&unbiased_int, sizeof(unbiased_int));

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
