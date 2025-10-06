/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Permute.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
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
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef permute_dims,
    const ValueRef out) {
  (void)permute_dims;
  VK_CHECK_COND(check_same_packed_dim(graph, in, out));

  // This implementation doesn't not requires the input tensor to have the same
  // dim size as the argument. The code will work as long as the input tensor's
  // dim size is shorter than the permute dim array. In this case, the code
  // assume size of 1 at the higher dimensions.
}

} // namespace

void resize_permute_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args[0].refs[0];
  const ValueRef in = args[1].refs[0];

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const std::vector<int64_t> out_sizes = graph->sizes_of(out);

  const std::vector<int64_t> permute_dims =
      graph->extract_int_or_symint_list(resize_args[0]);

  if (in_sizes.size() == out_sizes.size() &&
      in_sizes.size() == permute_dims.size()) {
    std::vector<int64_t> new_out_sizes(out_sizes.size(), 1);
    const int64_t out_ndim = std::max(in_sizes.size(), out_sizes.size());
    for (int i = 0; i < out_ndim; i++) {
      const int64_t permute_dim = permute_dims.at(i);
      new_out_sizes.at(i) = in_sizes.at(permute_dim);
    }
    graph->virtual_resize(out, new_out_sizes);
  }
  // Case where permute is being used to implement squeeze
  else if (
      in_sizes.size() > out_sizes.size() &&
      in_sizes.size() == permute_dims.size()) {
    std::vector<int64_t> new_out_sizes(out_sizes.size(), 1);
    const size_t offset = in_sizes.size() - out_sizes.size();
    for (int i = 0; i < out_sizes.size(); i++) {
      const int64_t permute_dim = permute_dims.at(i + offset);
      new_out_sizes.at(i) = in_sizes.at(permute_dim);
    }
    graph->virtual_resize(out, new_out_sizes);
  }
  // Case where Permute is being used to implement unsqueeze
  else if (
      in_sizes.size() < out_sizes.size() &&
      out_sizes.size() == permute_dims.size()) {
    std::vector<int64_t> new_out_sizes(out_sizes.size(), 1);
    const size_t offset = out_sizes.size() - in_sizes.size();
    for (int i = 0; i < out_sizes.size(); i++) {
      int64_t permute_dim = permute_dims.at(i) - offset;
      if (permute_dim >= 0) {
        new_out_sizes.at(i) = in_sizes.at(permute_dim);
      }
    }
    graph->virtual_resize(out, new_out_sizes);
  } else {
    VK_THROW("Invalid permute dims");
  }
}

void add_permute_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef permute_dims,
    const ValueRef out) {
  check_args(graph, in, permute_dims, out);

  // Convert the permute dims to WHCN dimension order, which is the standard in
  // our compute shaders. The following transformations are applied.
  // 1. Change dimension index values from NCHW order valueto WHCN order value
  // 2. Reverse the order of the permute array from NCHW order to WHCN order
  ivec4 whcn_permute_dims{0, 1, 2, 3};
  {
    IntListPtr permute_dims_ptr = graph.get_int_list(permute_dims);
    const int32_t permute_ndim =
        utils::safe_downcast<int32_t>(permute_dims_ptr->size());

    for (int32_t nchw_i = permute_ndim - 1, whcn_i = 0; nchw_i >= 0;
         nchw_i--, whcn_i++) {
      int32_t permute_dim_nchw =
          utils::safe_downcast<int32_t>(permute_dims_ptr->at(nchw_i));
      if (permute_dim_nchw < 0) {
        permute_dim_nchw += permute_ndim;
      }
      const int32_t permute_dim_whcn = permute_ndim - 1 - permute_dim_nchw;

      whcn_permute_dims[whcn_i] = permute_dim_whcn;
    }
  }

  std::string kernel_name = "permute";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_buffers;
  std::vector<PushConstantDataInfo> push_constants;
  vkapi::SpecVarList spec_vars;

  const int32_t out_channels = dim_at<kChannel4D>(graph.sizes_of(out));
  const int32_t in_channels = dim_at<kChannel4D>(graph.sizes_of(in));

  const int32_t packed_dim = graph.packed_dim_of(in);
  ivec2 channel_info = {out_channels, in_channels};
  if (packed_dim == WHCN::kChannelsDim) {
    channel_info[0] = utils::align_up_4(channel_info[0]);
    channel_info[1] = utils::align_up_4(channel_info[1]);
  }

  push_constants = {
      graph.sizes_pc_of(out),
      graph.sizes_pc_of(in),
      PushConstantDataInfo(&whcn_permute_dims, sizeof(whcn_permute_dims))};

  spec_vars = {graph.hashed_layout_of(out), graph.hashed_layout_of(in)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {permute_dims},
      // Resizing Logic
      resize_permute_node));
}

struct WHCNPermuteDims {
  int32_t whcn_permute_dims[api::kTensorDimLimit];

  void initialize(const std::vector<int64_t>& permute_dims) {
    const int32_t permute_ndim = permute_dims.size();
    for (int32_t whcn_i = 0; whcn_i < permute_ndim; whcn_i++) {
      const int32_t nchw_i = permute_ndim - 1 - whcn_i;
      int64_t index_val = permute_dims.at(nchw_i);
      if (index_val < 0) {
        index_val += permute_ndim;
      }
      const int32_t permute_dim_whcn = permute_ndim - 1 - index_val;
      whcn_permute_dims[whcn_i] = permute_dim_whcn;
    }
    for (int32_t whcn_i = permute_ndim; whcn_i < api::kTensorDimLimit;
         whcn_i++) {
      whcn_permute_dims[whcn_i] = whcn_i;
    }
  }
};

void add_permute_buffer_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef permute_dims,
    const ValueRef out) {
  check_args(graph, in, permute_dims, out);

  WHCNPermuteDims whcn_permute_dims;
  // Convert the permute dims to WHCN dimension order, which is the standard in
  // our compute shaders. The following transformations are applied.
  // 1. Change dimension index values from NCHW order valueto WHCN order value
  // 2. Extend the permute array to kTensorDimLimit
  {
    IntListPtr permute_dims_ptr = graph.get_int_list(permute_dims);
    whcn_permute_dims.initialize(*permute_dims_ptr);
  }

  std::string kernel_name = "permute";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_buffers = {
      graph.buffer_meta_ubo(out),
      graph.buffer_meta_ubo(in),
      graph.create_params_buffer(whcn_permute_dims),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {permute_dims},
      // Resizing Logic
      resize_permute_node));
}

void permute(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef in = args.at(idx++);
  const ValueRef permute_dims = args.at(idx++);
  const ValueRef out = args.at(idx++);

  if (graph.is_buffer_storage(args[2])) {
    return add_permute_buffer_node(graph, in, permute_dims, out);
  }
  return add_permute_node(graph, in, permute_dims, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.permute.default, permute);
  VK_REGISTER_OP(aten.permute_copy.default, permute);
}

} // namespace vkcompute
