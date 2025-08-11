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

void add_embedding_node(
    ComputeGraph& graph,
    ValueRef weight,
    ValueRef in,
    ValueRef out) {
  check_embedding_args(graph, weight, in, out);

  std::string kernel_name = "embedding";
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
  ValueRef in = args[1];
  ValueRef out = args[5];

  ValueRef weight = prepack_standard(
      graph,
      args[0],
      StorageType::TEXTURE_2D,
      GPUMemoryLayout::TENSOR_HEIGHT_PACKED);

  add_embedding_node(graph, weight, in, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.embedding.default, embedding);
}

} // namespace vkcompute
