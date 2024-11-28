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

#include <executorch/backends/vulkan/runtime/utils/StorageUtils.h>

namespace vkcompute {

using utils::GPUMemoryLayout;
using utils::StorageType;

void check_embedding_args(
    const api::vTensor& weight,
    const api::vTensor& in,
    const api::vTensor& out) {
  // The packing logic may not be trivial here. Input and output are Channel
  // Packed, which is default for the Vulkan backend. However, weight vector is
  // height-packed instead of channel-packed for space reason.
  VK_CHECK_COND(check_packed_dim_is(weight, WHCN::kHeightDim));
  VK_CHECK_COND(check_packed_dim_is(in, WHCN::kChannelsDim));
  VK_CHECK_COND(check_packed_dim_is(out, WHCN::kChannelsDim));
}

void add_embedding_node(
    ComputeGraph& graph,
    ValueRef weight,
    ValueRef in,
    ValueRef out) {
  vTensorPtr t_weight = graph.get_tensor(weight);
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  check_embedding_args(*t_weight, *t_in, *t_out);

  std::string kernel_name = "embedding";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      {{out, vkapi::kWrite}, {{in, weight}, vkapi::kRead}},
      {
          t_out->sizes_ubo(),
      },
      {t_out->hashed_layout(),
       t_in->hashed_layout(),
       t_weight->hashed_layout()}));
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
