/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Copy.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_cat_default_node(
    ComputeGraph& graph,
    ValueRef in_list_ref,
    ValueRef dim_ref,
    ValueRef out) {
  ValueListPtr input_list = graph.get_value_list(in_list_ref);

  for (ValueRef input_ref : *input_list) {
    vTensorPtr t_in = graph.get_tensor(input_ref);
    VK_CHECK_COND(check_memory_layout_is(*t_in, utils::kChannelsPacked));
  }

  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  vTensorPtr t_out = graph.get_tensor(out);

  DimIndex dim_index = normalize_to_dim_index(*t_out, dim);

  // TODO: Find ways to factor out the similar code for width, height, and batch
  if (dim_index == kWidth4D) {
    utils::ivec3 src_offset = utils::make_ivec3({0, 0, 0}, false);
    utils::ivec3 dst_offset = utils::make_ivec3({0, 0, 0}, false);

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      utils::ivec3 range = t_in->texture_limits();
      add_copy_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset[0] += range[0];
    }

  } else if (dim_index == kHeight4D) {
    utils::ivec3 src_offset = utils::make_ivec3({0, 0, 0}, false);
    utils::ivec3 dst_offset = utils::make_ivec3({0, 0, 0}, false);

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      utils::ivec3 range = t_in->texture_limits();
      add_copy_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset[1] += range[1];
    }
  } else if (dim_index == kBatch4D) {
    utils::ivec3 src_offset = utils::make_ivec3({0, 0, 0}, false);
    utils::ivec3 dst_offset = utils::make_ivec3({0, 0, 0}, false);

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      utils::ivec3 range = t_in->texture_limits();
      add_copy_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset[2] += range[2];
    }
  } else if (dim_index == kChannel4D) {
    int32_t src_offset = 0;
    int32_t dst_offset = 0;

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      int32_t range = dim_at(t_in->sizes(), kChannel4D);
      add_copy_channel_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset += range;
    }
  } else {
    VK_THROW("Unexpected value of dim_index=", dim_index);
  }
}

void cat_default(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_cat_default_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.cat.default, cat_default);
}

} // namespace vkcompute
