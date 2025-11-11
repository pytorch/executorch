/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/containers/PushConstantData.h>

namespace vkcompute {

uint32_t PushConstantDataInfo::write(
    void* dst,
    const uint32_t dst_offset,
    const uint32_t max_dst_size) const {
  if (tensorUniformData != nullptr) {
    return tensorUniformData->write_attribute(
        dst, dst_offset, max_dst_size, payload_.attr);
  }

  VK_CHECK_COND(
      (dst_offset + payload_.dataSize) <= max_dst_size,
      "Attempting to write push constant data outside data boundary.");
  memcpy((uint8_t*)dst + dst_offset, payload_.data, payload_.dataSize);
  return payload_.dataSize;
}

} // namespace vkcompute
