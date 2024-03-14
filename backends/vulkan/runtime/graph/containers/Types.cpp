/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/containers/Types.h>

namespace at {
namespace native {
namespace vulkan {

std::ostream& operator<<(std::ostream& out, const TypeTag& tag) {
  switch (tag) {
    case TypeTag::NONE:
      out << "NONE";
      break;
    case TypeTag::TENSOR:
      out << "TENSOR";
      break;
    case TypeTag::STAGING:
      out << "STAGING";
      break;
    default:
      out << "UNKNOWN";
      break;
  }
  return out;
}

} // namespace vulkan
} // namespace native
} // namespace at
