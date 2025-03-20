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
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  vTensorPtr t_out = graph.get_tensor(out);

  const auto packed_dim = t_out->packed_dim();
  const auto packed_dim_index = static_cast<DimIndex>(kWidth4D - packed_dim);

  DimIndex dim_index = normalize_to_dim_index(*t_out, dim);
  // Index of dimension to be concatenated in (w, h, c * b) coordinate system
  const auto dim_xyz_index = std::min(2, -dim_index - 1);

  if (dim_index > kWidth4D || dim_index < kBatch4D) {
    VK_THROW("Unexpected value of dim_index=", dim_index);
  }

  utils::ivec4 src_offset = utils::make_ivec4({0, 0, 0, 0}, false);
  utils::ivec4 dst_offset = utils::make_ivec4({0, 0, 0, 0}, false);

  const bool is_concat_channel = (dim_index == kChannel4D);

  // if concatenating channels
  if (is_concat_channel) {
    // set destination offset w as channel size of the output tensor
    dst_offset[3] = dim_at(t_out->sizes(), kChannel4D);
  }

  for (ValueRef input_ref : *input_list) {
    const vTensorPtr t_in = graph.get_tensor(input_ref);
    const utils::ivec3 range = t_in->logical_limits();
    const auto in_channel_size = dim_at(t_in->sizes(), kChannel4D);
    // if concatenating same dimension as the packed dimension
    if (dim_index == packed_dim_index) {
      // if concatenating channels, use add_copy_channel_offset_node function as
      // add_copy_packed_dim_offset_node does not support channel packing
      if (is_concat_channel) {
        add_copy_channel_offset_node(
            graph,
            input_ref,
            in_channel_size,
            src_offset[2],
            dst_offset[2],
            out);
        dst_offset[dim_xyz_index] += in_channel_size;
      } else {
        // src_offset[3] is not used now but will be used in the future when
        // add_copy_packed_dim_offset_node will support channel packing
        //
        // set source offset w as channel size of the output tensor if
        // concatenating channels
        src_offset[3] = is_concat_channel ? in_channel_size : 0;
        add_copy_packed_dim_offset_node(
            graph, input_ref, range, src_offset, dst_offset, out);
        dst_offset[dim_xyz_index] += dim_at(t_in->sizes(), packed_dim_index);
      }
    } else {
      // set source offset w as channel size of the output tensor if
      // concatenating channels
      src_offset[3] = is_concat_channel ? in_channel_size : 0;
      add_copy_offset_node(
          graph, input_ref, range, src_offset, dst_offset, out, true, false);
      dst_offset[dim_xyz_index] +=
          is_concat_channel ? in_channel_size : range[dim_xyz_index];
    }
  }
}

void cat_default(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_cat_default_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.cat.default, cat_default);
}

} // namespace vkcompute
