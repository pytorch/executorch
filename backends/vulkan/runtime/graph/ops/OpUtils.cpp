/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OpUtils.h>

namespace at {
namespace native {
namespace vulkan {

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

} // namespace vulkan
} // namespace native
} // namespace at
