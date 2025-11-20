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

#include <executorch/backends/vulkan/runtime/utils/StorageUtils.h>

namespace vkcompute {

using utils::GPUMemoryLayout;
using utils::StorageType;

void resize_split_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef input = args.at(0).refs.at(0);
  const ValueRef split_sizes_ref = args.at(1).refs.at(0);
  const ValueRef dim_ref = args.at(2).refs.at(0);
  const ValueRef out_list_ref = args.at(3).refs.at(0);

  const ValueListPtr out_list = graph->get_value_list(out_list_ref);
  const std::vector<int64_t> split_sizes =
      *(graph->get_int_list(split_sizes_ref));
  const int64_t dim = graph->extract_scalar<int64_t>(dim_ref);

  const int64_t input_ndim = graph->dim_of(input);
  const DimIndex dim_index = dim < 0 ? static_cast<DimIndex>(dim)
                                     : static_cast<DimIndex>(dim - input_ndim);

  std::vector<int64_t> input_sizes = graph->sizes_of(input);

  for (int split_idx = 0; split_idx < split_sizes.size(); split_idx++) {
    const int64_t split_size = split_sizes.at(split_idx);
    const ValueRef out_ref = out_list->at(split_idx);

    std::vector<int64_t> out_sizes = input_sizes;
    out_sizes.at(dim_index) = split_size;

    graph->virtual_resize(out_ref, out_sizes);
  }
}

void add_split_node(
    ComputeGraph& graph,
    const ValueRef input,
    const std::vector<int64_t>& split_sizes,
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
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {utils::safe_downcast<int32_t>(dim_whcn),
       static_cast<int32_t>(split_idx),
       static_cast<int32_t>(split_offset)},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void add_split_with_sizes_node(
    ComputeGraph& graph,
    const ValueRef input,
    const std::vector<int64_t>& split_sizes,
    const int64_t dim,
    const ValueRef out_list_ref) {
  const ValueListPtr out_list = graph.get_value_list(out_list_ref);

  VK_CHECK_COND(out_list->size() == split_sizes.size());

  // Dispatch a shader for each output tensor
  for (int split_idx = 0; split_idx < split_sizes.size(); split_idx++) {
    const ValueRef out_ref = out_list->at(split_idx);
    add_split_node(graph, input, split_sizes, dim, out_ref, split_idx);
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
  std::vector<int64_t> split_sizes = *(graph.get_int_list(split_sizes_ref));

  add_split_with_sizes_node(graph, input, split_sizes, dim, out_list_ref);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      aten.split_with_sizes_copy.default, split_with_sizes_copy_default);
}

} // namespace vkcompute
