/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/QuantizationConfig.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/utils/StorageUtils.h>

namespace vkcompute {

utils::uvec3 pick_embedding_q4gsw_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<int64_t>& sizes = graph->sizes_of(out);
  int ndim = sizes.size();

  uint32_t blocks_per_row = static_cast<uint32_t>(sizes[ndim - 1]) / 32;
  uint32_t height = ndim >= 2 ? static_cast<uint32_t>(sizes[ndim - 2]) : 1;
  uint32_t depth = 1;
  for (int i = 0; i < ndim - 2; i++) {
    depth *= static_cast<uint32_t>(sizes[i]);
  }

  return {blocks_per_row, height, depth};
}

void resize_embedding_q4gsw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef indices = args.at(1).refs.at(0);

  const int64_t embed_dim = graph->get_int(resize_args.at(0));
  const std::vector<int64_t> indices_sizes = graph->sizes_of(indices);

  // Output shape is indices.shape + [embed_dim]
  std::vector<int64_t> out_sizes = indices_sizes;
  out_sizes.push_back(embed_dim);

  graph->virtual_resize(out, out_sizes);
}

void add_embedding_q4gsw_node(
    ComputeGraph& graph,
    const ValueRef indices,
    const ValueRef weight,
    const ValueRef weight_scales,
    const int32_t group_size,
    const int32_t embed_dim,
    const int32_t num_indices,
    const int32_t out_height,
    const int32_t is_linear_weight,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(indices) == WHCN::kWidthDim);
  VK_CHECK_COND(embed_dim % 32 == 0, "embed_dim must be a multiple of 32");

  std::string kernel_name = "embedding_q4gsw";
  kernel_name.reserve(kShaderNameReserve);

  vkapi::ScalarType scales_dtype = graph.dtype_of(weight_scales);
  if (scales_dtype != vkapi::kHalf) {
    kernel_name += "_float_scales";
  }

  if (is_linear_weight) {
    kernel_name += "_linear_weight";
    add_storage_type_suffix(kernel_name, graph.storage_type_of(weight));
  }

  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&group_size, sizeof(group_size)),
      PushConstantDataInfo(&embed_dim, sizeof(embed_dim)),
      PushConstantDataInfo(&num_indices, sizeof(num_indices)),
      PushConstantDataInfo(&out_height, sizeof(out_height)),
      PushConstantDataInfo(&is_linear_weight, sizeof(is_linear_weight)),
  };

  ValueRef embed_dim_ref = graph.add_scalar<int64_t>(embed_dim);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_embedding_q4gsw_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {{indices, weight, weight_scales}, vkapi::kRead}},
      {},
      push_constants,
      {},
      {embed_dim_ref},
      resize_embedding_q4gsw_node));
}

void embedding_q4gsw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef weight_data = args[0];
  ValueRef weight_scales_data = args[1];
  ValueRef group_size_ref = args[2];
  ValueRef indices = args[3];
  ValueRef is_linear_weight_ref = args[4];
  ValueRef out = args[5];

  int32_t group_size = graph.extract_scalar<int32_t>(group_size_ref);
  int32_t is_linear_weight =
      graph.extract_scalar<bool>(is_linear_weight_ref) ? 1 : 0;

  const std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  int32_t embed_dim = static_cast<int32_t>(weight_sizes.back() * 2);

  const std::vector<int64_t> indices_sizes = graph.sizes_of(indices);
  int32_t num_indices = 1;
  for (auto s : indices_sizes) {
    num_indices *= static_cast<int32_t>(s);
  }
  int32_t out_height = static_cast<int32_t>(indices_sizes.back());

  ValueRef weight;
  if (is_linear_weight) {
    QuantizationConfig weight_quant_config(4, kPerGroup, {group_size});
    weight = prepack_quantized_linear_weight(
        graph, weight_quant_config, weight_data);
  } else {
    weight = prepack_standard(
        graph, weight_data, utils::kBuffer, utils::kWidthPacked);
  }
  ValueRef weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  add_embedding_q4gsw_node(
      graph,
      indices,
      weight,
      weight_scales,
      group_size,
      embed_dim,
      num_indices,
      out_height,
      is_linear_weight,
      out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.embedding_q4gsw.default, embedding_q4gsw);
}

} // namespace vkcompute
