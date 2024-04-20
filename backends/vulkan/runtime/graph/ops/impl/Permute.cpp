/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using api::utils::ivec3;
using api::utils::uvec2;
using api::utils::uvec4;

void check_args(
    const vTensor& in,
    const IntListPtr& permute_dims,
    const vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(in, api::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, api::kChannelsPacked));

  int64_t in_dim = in.dim();
  VK_CHECK_COND(
      in_dim == permute_dims->size(),
      "Input tensor dim size must match argument");
}

void add_permute_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef permute_dims_ref,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  IntListPtr permute_dims = graph.get_int_list(permute_dims_ref);

  check_args(*t_in, permute_dims, *t_out);

  uvec4 in_size{1u, 1u, 1u, 1u}, out_size{1u, 1u, 1u, 1u};
  uvec4 out_dims{0u, 1u, 2u, 3u};

  int64_t in_dim = t_in->dim();

  std::vector<bool> seen(in_dim);
  for (int i = 0; i < in_dim; i++) {
    int64_t permute_dim = (*permute_dims)[i];
    VK_CHECK_COND(
        !seen[permute_dim], "Argument dim ", permute_dim, "  is repeated");
    seen[permute_dim] = true;

    // Map to 4D tensor dims.
    in_size.data[(4u - in_dim) + i] = t_in->size(i);
    out_size.data[(4u - in_dim) + i] = t_in->size(permute_dim);
    out_dims.data[(4u - in_dim) + i] = permute_dim + (4u - in_dim);
  }

  std::string kernel_name = "permute";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  uint32_t out_channels = out_size.data[1u];
  uint32_t in_channels = in_size.data[1u];

  uint32_t out_c_aligned = api::utils::align_up(out_channels, 4u);
  uint32_t in_c_aligned = api::utils::align_up(in_channels, 4u);

  const struct Block final {
    uvec4 out_ndims;
    uvec2 ch_info;
  } params{
      out_dims,
      {out_c_aligned, in_c_aligned},
  };

  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      {{out, api::MemoryAccessType::WRITE}, {in, api::MemoryAccessType::READ}},
      {t_out->sizes_ubo(), graph.create_params_buffer(params)},
      // Specialization Constants
      {SV(t_out->gpu_memory_layout_int())},
      // Resizing Logic
      nullptr,
      {}));
}

void permute(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_permute_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.permute.default, permute);
  VK_REGISTER_OP(aten.permute_copy.default, permute);
}

} // namespace vkcompute
