/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>

namespace vkcompute {

//
// Tensor output size calculation functions
//

std::vector<int64_t> calculate_broadcasted_output_size(
    const vTensor& t1,
    const vTensor& t2) {
  std::vector<int64_t> out_sizes(
      std::max(t1.sizes().size(), t2.sizes().size()));

  // Match the sizes in reverse because sizes are in NCHW order
  for (int i = -1; i >= -out_sizes.size(); --i) {
    out_sizes.at(out_sizes.size() + i) = std::max(
        api::utils::val_at(i, t1.sizes()), api::utils::val_at(i, t2.sizes()));
  }

  return out_sizes;
}

//
// Tensor property checking functions
//

bool check_ndim_is(const vTensor& t, size_t ndim) {
  return t.sizes().size() == ndim;
}

bool check_same_sizes_at(
    const vTensor& t1,
    const int64_t d1,
    const vTensor& t2,
    const int64_t d2) {
  return api::utils::val_at(d1, t1.sizes()) ==
      api::utils::val_at(d2, t2.sizes());
}

bool check_memory_layout_is(const vTensor& t, api::GPUMemoryLayout layout) {
  return t.gpu_memory_layout() == layout;
}

bool check_same_ndim(const vTensor& t1, const vTensor& t2) {
  return t1.sizes().size() == t2.sizes().size();
}

bool check_same_memory_layout(const vTensor& t1, const vTensor& t2) {
  return t1.gpu_memory_layout() == t2.gpu_memory_layout();
}

bool check_same_memory_layout(
    const vTensor& t1,
    const vTensor& t2,
    const vTensor& t3) {
  if (t1.gpu_memory_layout() != t2.gpu_memory_layout()) {
    return false;
  }
  return (t1.gpu_memory_layout() == t3.gpu_memory_layout());
}

//
// Broadcast flag functions
//

bool is_packed_dim_broadcasted(const vTensor& sndr, const vTensor& rcvr) {
  // We assume that the tensors are broadcastable. If values aren't equal at
  // some index, then the value of rcvr is 1 and hence should be broadcasted.
  switch (sndr.gpu_memory_layout()) {
    case api::kChannelsPacked:
      return api::utils::val_at(-3, sndr.sizes()) >
          api::utils::val_at(-3, rcvr.sizes());
    case api::kHeightPacked:
      return api::utils::val_at(-2, sndr.sizes()) >
          api::utils::val_at(-2, rcvr.sizes());
    case api::kWidthPacked:
      return api::utils::val_at(-1, sndr.sizes()) >
          api::utils::val_at(-1, rcvr.sizes());
  }
}

api::utils::ivec2 create_broadcast_params(
    const vTensor& t1,
    const vTensor& t2) {
  return api::utils::make_ivec2(
      {is_packed_dim_broadcasted(t2, t1), is_packed_dim_broadcasted(t1, t2)});
}

//
// Work group size calculation functions
//

api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group) {
  api::utils::uvec3 local_group_size = {4, 4, 4};
  if (global_work_group.data[2u] == 1) {
    if (global_work_group.data[1u] < 8) {
      local_group_size.data[0u] = 16;
      local_group_size.data[1u] = 4;
      local_group_size.data[2u] = 1;
    } else {
      local_group_size.data[0u] = 8;
      local_group_size.data[1u] = 8;
      local_group_size.data[2u] = 1;
    }
  }
  return local_group_size;
}

} // namespace vkcompute
