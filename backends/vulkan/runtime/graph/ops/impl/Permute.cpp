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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using utils::ivec4;

namespace {

void check_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef permute_dims,
    const ValueRef out) {
  (void)permute_dims;
  VK_CHECK_COND(check_same_packed_dim(graph, in, out));
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

  int32_t pack_into_int32() const {
    VK_CHECK_COND(api::kTensorDimLimit <= 8);
    int32_t packed = 0;
    for (int32_t i = 0; i < api::kTensorDimLimit; i++) {
      packed |= (whcn_permute_dims[i] & 0x0F) << (i * 4);
    }
    return packed;
  }
};

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

  std::string kernel_name = "permute";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out), graph.meta_ubo(in)};

  std::vector<PushConstantDataInfo> push_constants;
  vkapi::SpecVarList spec_vars = {
      graph.hashed_layout_of(out), graph.hashed_layout_of(in)};

  // WHCN permute dims for the texture path (ivec4, max 4D).
  // Declared here so its lifetime extends to the DynamicDispatchNode creation
  // where push_constants references it.
  ivec4 whcn_permute_dims{0, 1, 2, 3};

  if (graph.is_buffer_storage(out)) {
    // Buffer path: supports up to kTensorDimLimit dims via WHCNPermuteDims,
    // packed into a spec constant int
    WHCNPermuteDims whcn_pd;
    whcn_pd.initialize(*graph.get_int_list(permute_dims));
    spec_vars.append(whcn_pd.pack_into_int32());
  } else {
    // Texture path: compute 4D WHCN permute dims and pass as push constant
    IntListPtr permute_dims_ptr = graph.get_int_list(permute_dims);
    const int32_t permute_ndim =
        utils::safe_downcast<int32_t>(permute_dims_ptr->size());
    VK_CHECK_COND(
        permute_ndim <= 4,
        "Texture storage only supports permute with up to 4 dims");

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

    push_constants.push_back(
        PushConstantDataInfo(&whcn_permute_dims, sizeof(whcn_permute_dims)));
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      param_ubos,
      push_constants,
      spec_vars,
      {permute_dims},
      resize_permute_node));
}

void permute(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef in = args.at(idx++);
  const ValueRef permute_dims = args.at(idx++);
  const ValueRef out = args.at(idx++);

  add_permute_node(graph, in, permute_dims, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.permute.default, permute);
  VK_REGISTER_OP(aten.permute_copy.default, permute);
}

} // namespace vkcompute
