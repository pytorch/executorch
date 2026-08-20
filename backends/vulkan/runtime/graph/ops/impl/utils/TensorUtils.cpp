/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

//
// Tensor output size calculation functions
//

std::vector<int64_t> calculate_broadcasted_output_size(
    const std::vector<int64_t>& sizes1,
    const std::vector<int64_t>& sizes2) {
  std::vector<int64_t> out_sizes(std::max(sizes1.size(), sizes2.size()));

  // Match the sizes in reverse because sizes are in NCHW order
  for (int i = -1; i >= -out_sizes.size(); --i) {
    out_sizes.at(out_sizes.size() + i) =
        std::max(utils::val_at(i, sizes1), utils::val_at(i, sizes2));
  }

  return out_sizes;
}

//
// Tensor property checking functions
//

bool check_same_packed_dim(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  return graph.packed_dim_of(in) == graph.packed_dim_of(out);
}

//
// Broadcast flag functions
//

bool is_packed_dim_broadcasted(
    ComputeGraph& graph,
    const ValueRef sndr,
    const ValueRef rcvr) {
  // We assume that the tensors are broadcastable. If values aren't equal at
  // some index, then the value of rcvr is 1 and hence should be broadcasted.
  const std::vector<int64_t> sndr_sizes = graph.sizes_of(sndr);
  const std::vector<int64_t> rcvr_sizes = graph.sizes_of(rcvr);

  switch (graph.packed_dim_of(sndr)) {
    case WHCN::kChannelsDim:
      return utils::val_at(-3, sndr_sizes) > utils::val_at(-3, rcvr_sizes);
    case WHCN::kHeightDim:
      return utils::val_at(-2, sndr_sizes) > utils::val_at(-2, rcvr_sizes);
    case WHCN::kWidthDim:
      return utils::val_at(-1, sndr_sizes) > utils::val_at(-1, rcvr_sizes);
    default:
      VK_THROW("Invalid packed dim");
  }
}

utils::ivec2 create_broadcast_params(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2) {
  return utils::make_ivec2(
      {is_packed_dim_broadcasted(graph, t2, t1),
       is_packed_dim_broadcasted(graph, t1, t2)});
}

//
// Work group size calculation functions
//

utils::uvec3 adaptive_work_group_size(const utils::uvec3& global_work_group) {
  utils::uvec3 local_group_size = {4, 4, 4};
  if (global_work_group[2u] == 1) {
    if (global_work_group[1u] < 8) {
      local_group_size[0u] = 16;
      local_group_size[1u] = 4;
      local_group_size[2u] = 1;
    } else {
      local_group_size[0u] = 8;
      local_group_size[1u] = 8;
      local_group_size[2u] = 1;
    }
  }
  return local_group_size;
}

} // namespace vkcompute
