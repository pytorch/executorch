/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_index_select_args(
    const api::vTensor& in,
    const api::vTensor& idx,
    const api::vTensor& out) {
  VK_CHECK_COND(check_packed_dim_is(in, WHCN::kChannelsDim));
  VK_CHECK_COND(check_packed_dim_is(idx, WHCN::kChannelsDim));
  VK_CHECK_COND(check_packed_dim_is(out, WHCN::kChannelsDim));
}

void add_index_select_channel_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef idx,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_idx = graph.get_tensor(idx);
  vTensorPtr t_out = graph.get_tensor(out);

  check_index_select_args(*t_in, *t_idx, *t_out);

  std::string kernel_name = "index_select_channel";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      {{out, vkapi::kWrite}, {{in, idx}, vkapi::kRead}},
      {t_out->sizes_ubo(), t_in->sizes_ubo()},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

struct IndexSelectParams final {
  int32_t gpu_dim;
  int32_t stride;
};

IndexSelectParams create_index_select_params(
    const int64_t dim_idx,
    const api::vTensor& in) {
  if (dim_idx == kWidth4D) {
    return {0, 1};
  } else if (dim_idx == kHeight4D) {
    return {1, 1};
  } else if (dim_idx == kBatch4D) {
    int64_t n_channels = dim_at(in.sizes(), kChannel4D);
    int64_t stride = utils::div_up_4(n_channels);
    return {2, static_cast<int32_t>(stride)};
  } else {
    VK_THROW("Unexpected dim_idx!");
  }
}

void add_index_select_node(
    ComputeGraph& graph,
    ValueRef in,
    const int64_t dim_idx,
    ValueRef idx,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_idx = graph.get_tensor(idx);
  vTensorPtr t_out = graph.get_tensor(out);

  check_index_select_args(*t_in, *t_idx, *t_out);

  IndexSelectParams params = create_index_select_params(dim_idx, *t_in);

  std::string kernel_name = "index_select";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      {{out, vkapi::kWrite}, {{in, idx}, vkapi::kRead}},
      {t_out->sizes_ubo(), graph.create_params_buffer(params)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

int64_t get_dim_idx(ComputeGraph& graph, ValueRef in, ValueRef dim_ref) {
  vTensorPtr t_in = graph.get_tensor(in);
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  dim = normalize(dim, t_in->dim());
  return normalize_to_dim_index(*t_in, dim);
}

void index_select(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef in = args[0];
  ValueRef dim_ref = args[1];
  ValueRef idx = args[2];
  ValueRef out = args[3];

  const int64_t dim_idx = get_dim_idx(graph, in, dim_ref);
  if (dim_idx == kChannel4D) {
    add_index_select_channel_node(graph, in, idx, out);
  } else {
    add_index_select_node(graph, in, dim_idx, idx, out);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.index_select.default, index_select);
}

} // namespace vkcompute
