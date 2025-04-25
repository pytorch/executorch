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

using utils::ivec2;
using utils::ivec3;
using utils::ivec4;
using utils::uvec4;

namespace {

void check_args(
    const api::vTensor& in,
    const std::vector<int64_t>& permute_dims,
    const api::vTensor& out) {
  VK_CHECK_COND(check_same_packed_dim(in, out));

  // This implementation doesn't not requires the input tensor to have the same
  // dim size as the argument. The code will work as long as the input tensor's
  // dim size is shorter than the permute dim array. In this case, the code
  // assume size of 1 at the higher dimensions.
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

  // Special cases of squeeze/unsqueeze. Because the input dim size can be
  // different with output dim size. So pick t_in->dim() if squeeze, and
  // t_out->dim() if unsqueeze to create parameter for permute.
  int64_t out_ndim = std::max(t_in->dim(), t_out->dim());
  std::vector<bool> seen(out_ndim);
  for (int i = 0; i < out_ndim; i++) {
    int64_t permute_dim = permute_dims[i];
    VK_CHECK_COND(
        !seen[permute_dim], "Argument dim ", permute_dim, "  is repeated");
    seen[permute_dim] = true;

    out_dims[(4u - out_ndim) + i] = permute_dim + (4 - out_ndim);
  }

  std::string kernel_name = "permute";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  int32_t out_channels = dim_at<kChannel4D>(t_out->sizes());
  int32_t in_channels = dim_at<kChannel4D>(t_in->sizes());

  const auto packed_dim = graph.packed_dim_of(in);
  ivec2 channel_info = {out_channels, in_channels};
  if (packed_dim == WHCN::kChannelsDim) {
    channel_info[0] = utils::align_up_4(channel_info[0]);
    channel_info[1] = utils::align_up_4(channel_info[1]);
  }

  const vkapi::SpecVarList spec_vars = {packed_dim};

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      {},
      // Specialization Constants
      spec_vars,
      // Resizing Logic
      nullptr,
      {},
      {{graph.logical_limits_pc_of(out),
        graph.sizes_pc_of(in),
        PushConstantDataInfo(&out_dims, sizeof(out_dims)),
        PushConstantDataInfo(&channel_info, sizeof(channel_info))}}));
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
