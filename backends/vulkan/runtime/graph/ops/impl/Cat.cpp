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
    VK_CHECK_COND(check_memory_layout_is(*t_in, api::kChannelsPacked));
  }

  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  vTensorPtr t_out = graph.get_tensor(out);

  NchwDim nchw_dim = normalize_to_nchw_dim(*t_out, dim);

  // TODO: Find ways to factor out the similar code for width, height, and batch
  if (nchw_dim == DimWidth) {
    api::utils::ivec3 src_offset = api::utils::make_ivec3({0, 0, 0}, false);
    api::utils::ivec3 dst_offset = api::utils::make_ivec3({0, 0, 0}, false);

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      api::utils::ivec3 range = t_in->texture_limits();
      add_copy_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset.data[0] += range.data[0];
    }

  } else if (nchw_dim == DimHeight) {
    api::utils::ivec3 src_offset = api::utils::make_ivec3({0, 0, 0}, false);
    api::utils::ivec3 dst_offset = api::utils::make_ivec3({0, 0, 0}, false);

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      api::utils::ivec3 range = t_in->texture_limits();
      add_copy_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset.data[1] += range.data[1];
    }
  } else if (nchw_dim == DimBatch) {
    api::utils::ivec3 src_offset = api::utils::make_ivec3({0, 0, 0}, false);
    api::utils::ivec3 dst_offset = api::utils::make_ivec3({0, 0, 0}, false);

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      api::utils::ivec3 range = t_in->texture_limits();
      add_copy_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset.data[2] += range.data[2];
    }
  } else if (nchw_dim == DimChannel) {
    int32_t src_offset = 0;
    int32_t dst_offset = 0;

    for (ValueRef input_ref : *input_list) {
      vTensorPtr t_in = graph.get_tensor(input_ref);
      int32_t range = dim_at<Dim4D::Channel>(t_in->sizes());
      add_copy_channel_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out);
      dst_offset += range;
    }
  } else {
    VK_THROW("Unexpected value of nchw_dim=", nchw_dim);
  }
}

void cat_default(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_cat_default_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.cat.default, cat_default);
}

} // namespace vkcompute
