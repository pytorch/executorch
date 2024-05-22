/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Permute.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using api::utils::ivec2;
using api::utils::ivec3;
using api::utils::ivec4;
using api::utils::uvec4;

namespace {

void check_args(
    const vTensor& in,
    const std::vector<int64_t>& permute_dims,
    const vTensor& out) {
  VK_CHECK_COND(check_memory_layout_is(in, api::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, api::kChannelsPacked));

  // This implementation doesn't not requires the input tensor to have the same
  // dim size as the argument. The code will work as long as the input tensor's
  // dim size is shorter than the permute dim array. In this case, the code
  // assume size of 1 at the higher dimensions.

  int64_t out_dim = out.dim();
  VK_CHECK_COND(
      out_dim == permute_dims.size(),
      "Output tensor dim size must match argument");
}

} // namespace

void add_permute_node(
    ComputeGraph& graph,
    ValueRef in,
    const std::vector<int64_t>& permute_dims,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  check_args(*t_in, permute_dims, *t_out);

  ivec4 out_dims{0, 1, 2, 3};

  int64_t out_dim = t_out->dim();
  std::vector<bool> seen(out_dim);
  for (int i = 0; i < t_out->dim(); i++) {
    int64_t permute_dim = permute_dims[i];
    VK_CHECK_COND(
        !seen[permute_dim], "Argument dim ", permute_dim, "  is repeated");
    seen[permute_dim] = true;

    out_dims.data[(4u - out_dim) + i] = permute_dim + (4 - out_dim);
  }

  std::string kernel_name = "permute";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  int32_t out_channels = dim_at<kChannel4D>(t_out->sizes());
  int32_t in_channels = dim_at<kChannel4D>(t_in->sizes());

  int32_t out_c_aligned = api::utils::align_up(out_channels, 4);
  int32_t in_c_aligned = api::utils::align_up(in_channels, 4);

  const struct Block final {
    ivec4 out_ndims;
    ivec2 ch_info;
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
      {t_out->texture_limits_ubo(),
       t_out->sizes_ubo(),
       graph.create_params_buffer(params)},
      // Specialization Constants
      {},
      // Resizing Logic
      nullptr,
      {}));
}

void add_permute_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef permute_dims_ref,
    ValueRef out) {
  IntListPtr permute_dims = graph.get_int_list(permute_dims_ref);

  add_permute_node(graph, in, *permute_dims, out);
}

void permute(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_permute_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.permute.default, permute);
  VK_REGISTER_OP(aten.permute_copy.default, permute);
}

} // namespace vkcompute
