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

void check_embedding_args(
    ComputeGraph& graph,
    const ValueRef weight,
    const ValueRef in,
    const ValueRef out) {
  // The packing logic may not be trivial here. Input and output are Channel
  // Packed, which is default for the Vulkan backend. However, weight vector is
  // height-packed instead of channel-packed for space reason.
  VK_CHECK_COND(graph.packed_dim_of(weight) == WHCN::kHeightDim);
  VK_CHECK_COND(graph.packed_dim_of(in) == WHCN::kChannelsDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kChannelsDim);
}

void resize_embedding_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef weight = args.at(1).refs.at(0);
  const ValueRef indices = args.at(1).refs.at(1);

  const std::vector<int64_t> weight_sizes = graph->sizes_of(weight);
  const std::vector<int64_t> indices_sizes = graph->sizes_of(indices);

  // Output shape is indices.shape + [embedding_dim]
  // where embedding_dim is the last dimension of weight
  std::vector<int64_t> out_sizes = indices_sizes;
  out_sizes.push_back(weight_sizes.back());

  graph->virtual_resize(out, out_sizes);
}

void add_embedding_node(
    ComputeGraph& graph,
    const ValueRef indices,
    const ValueRef weight,
    const ValueRef out) {
  std::string kernel_name = "embedding";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out), graph.meta_ubo(indices), graph.meta_ubo(weight)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{indices, weight}, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_embedding_node));
}

void add_embedding_legacy_node(
    ComputeGraph& graph,
    ValueRef weight,
    ValueRef in,
    ValueRef out) {
  check_embedding_args(graph, weight, in, out);

  std::string kernel_name = "embedding_legacy";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {{in, weight}, vkapi::kRead}},
      {
          graph.sizes_ubo(out),
      },
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(in),
       graph.hashed_layout_of(weight)},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void embedding(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef weight_data = args[0];
  ValueRef indices = args[1];
  ValueRef out = args[5];

  // Legacy implementation that accepts channels packed texture tensors for
  // input/output. Needed to support some old models still in circulation.
  if (graph.is_standard_channels_packed_texture_tensor(indices)) {
    ValueRef weight = prepack_standard(
        graph, weight_data, utils::kTexture2D, utils::kHeightPacked);

    add_embedding_legacy_node(graph, weight, indices, out);
    return;
  }

  ValueRef weight =
      prepack_standard(graph, weight_data, utils::kBuffer, utils::kWidthPacked);

  // New implementation for contiguous buffer and width-packed texture tensors
  add_embedding_node(graph, indices, weight, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.embedding.default, embedding);
}

} // namespace vkcompute
