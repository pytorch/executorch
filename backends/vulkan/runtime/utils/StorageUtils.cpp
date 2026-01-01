/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/utils/StorageUtils.h>

namespace vkcompute {
namespace utils {

bool is_packed_int8_layout(const GPUMemoryLayout layout) {
  switch (layout) {
    case kPackedInt8_4W:
    case kPackedInt8_4C:
    case kPackedInt8_4H:
    case kPackedInt8_4W4C:
    case kPackedInt8_4H4W:
      return true;
    default:
      return false;
  }
}

} // namespace utils
} // namespace vkcompute
