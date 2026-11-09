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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_index_select_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef idx,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(in) == WHCN::kChannelsDim);
  VK_CHECK_COND(graph.packed_dim_of(idx) == WHCN::kChannelsDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kChannelsDim);
}

// index_select replaces the selected dim with as many entries as the index
// tensor holds and leaves every other dim alone.
std::vector<int64_t> index_select_out_sizes(
    ComputeGraph* graph,
    const ValueRef in,
    const ValueRef idx,
    const DimIndex dim_idx) {
  std::vector<int64_t> out_sizes = graph->sizes_of(in);
  const int64_t ndim = static_cast<int64_t>(out_sizes.size());
  // dim_idx is a negative index counted from the innermost dim.
  const int64_t dim = ndim + dim_idx;
  VK_CHECK_COND(dim >= 0 && dim < ndim);
  out_sizes.at(dim) = graph->numel_of(idx);
  return out_sizes;
}

void resize_index_select_channel_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef idx = args.at(1).refs.at(1);

  graph->virtual_resize(
      out, index_select_out_sizes(graph, in, idx, kChannel4D));
}

void add_index_select_channel_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef idx,
    ValueRef out) {
  check_index_select_args(graph, in, idx, out);

  std::string kernel_name = "index_select_channel";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      {{out, vkapi::kWrite}, {{in, idx}, vkapi::kRead}},
      {graph.sizes_ubo(out), graph.sizes_ubo(in)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_index_select_channel_node));
}

struct IndexSelectParams final {
  int32_t gpu_dim;
};

IndexSelectParams create_index_select_params(const int64_t dim_idx) {
  if (dim_idx == kWidth4D) {
    return {0};
  } else if (dim_idx == kHeight4D) {
    return {1};
  } else if (dim_idx == kBatch4D) {
    // The batch axis shares the z axis with the channels, so the shader steps
    // over one batch in units of channel texels. That stride is derived from
    // the channel count, which a resize can change, so the shader reads it out
    // of in_sizes rather than taking a value frozen at build time.
    return {2};
  } else {
    VK_THROW("Unexpected dim_idx!");
  }
}

void resize_index_select_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef idx = args.at(1).refs.at(1);

  const DimIndex dim_idx =
      static_cast<DimIndex>(graph->extract_scalar<int32_t>(resize_args.at(0)));

  graph->virtual_resize(out, index_select_out_sizes(graph, in, idx, dim_idx));
}

void add_index_select_node(
    ComputeGraph& graph,
    ValueRef in,
    const int64_t dim_idx,
    ValueRef idx,
    ValueRef out) {
  check_index_select_args(graph, in, idx, out);

  IndexSelectParams params = create_index_select_params(dim_idx);

  std::string kernel_name = "index_select";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      {{out, vkapi::kWrite}, {{in, idx}, vkapi::kRead}},
      {graph.sizes_ubo(out),
       graph.sizes_ubo(in),
       graph.create_params_buffer(params)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {graph.get_or_add_value_for_int(dim_idx)},
      // Resizing Logic
      resize_index_select_node));
}

int64_t get_dim_idx(ComputeGraph& graph, ValueRef in, ValueRef dim_ref) {
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  const int64_t ndim = graph.dim_of(in);
  dim = normalize(dim, ndim);

  // Convert to DimIndex - this replicates normalize_to_dim_index logic
  return dim < 0 ? dim : dim - ndim;
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
