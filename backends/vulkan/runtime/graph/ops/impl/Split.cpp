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

void resize_split_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef input = args.at(1).refs.at(0);
  const ValueRef split_sizes_ref = extra_args.at(0);
  const int64_t dim = graph->extract_scalar<int64_t>(extra_args.at(1));
  const int64_t split_idx = graph->extract_scalar<int64_t>(extra_args.at(2));

  std::vector<int64_t> split_sizes =
      graph->extract_int_or_symint_list(split_sizes_ref);

  std::vector<int64_t> out_sizes = graph->sizes_of(input);
  int64_t ndim = static_cast<int64_t>(out_sizes.size());
  int64_t normalized_dim = (dim % ndim + ndim) % ndim;
  out_sizes.at(normalized_dim) = split_sizes.at(split_idx);

  graph->virtual_resize(out, out_sizes);
}

void add_split_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef split_sizes_ref,
    const std::vector<int64_t>& split_sizes,
    const ValueRef dim_ref,
    const int64_t dim,
    const ValueRef out,
    const int split_idx) {
  std::string kernel_name = "split";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out), graph.meta_ubo(input)};

  int64_t dim_whcn = nchw_dim_to_whcn_dim(dim, graph.dim_of(input));

  // Calculate the offset for this split by summing previous split sizes
  int64_t split_offset = 0;
  for (int i = 0; i < split_idx; i++) {
    split_offset += split_sizes[i];
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {utils::safe_downcast<int32_t>(dim_whcn),
       static_cast<int32_t>(split_idx),
       static_cast<int32_t>(split_offset),
       graph.hashed_layout_of(out),
       graph.hashed_layout_of(input)},
      // Resize Args
      {split_sizes_ref, dim_ref, graph.add_scalar<int64_t>(split_idx)},
      // Resizing Logic
      resize_split_node));
}

void add_split_with_sizes_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef split_sizes_ref,
    const std::vector<int64_t>& split_sizes,
    const ValueRef dim_ref,
    const int64_t dim,
    const ValueRef out_list_ref) {
  const std::vector<ValueRef> outs = *graph.get_value_list(out_list_ref);

  VK_CHECK_COND(outs.size() == split_sizes.size());

  // Dispatch a shader for each output tensor
  for (int split_idx = 0; split_idx < split_sizes.size(); split_idx++) {
    const ValueRef out_ref = outs.at(split_idx);
    add_split_node(
        graph,
        input,
        split_sizes_ref,
        split_sizes,
        dim_ref,
        dim,
        out_ref,
        split_idx);
  }
}

void split_with_sizes_copy_default(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  ValueRef input = args[0];
  ValueRef split_sizes_ref = args[1];
  ValueRef dim_ref = args[2];
  ValueRef out_list_ref = args[3];

  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  std::vector<int64_t> split_sizes =
      graph.extract_int_or_symint_list(split_sizes_ref);

  add_split_with_sizes_node(
      graph, input, split_sizes_ref, split_sizes, dim_ref, dim, out_list_ref);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      aten.split_with_sizes_copy.default, split_with_sizes_copy_default);
}

} // namespace vkcompute
