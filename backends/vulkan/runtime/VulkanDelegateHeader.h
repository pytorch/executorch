/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/result.h>

namespace executorch {
namespace backends {
namespace vulkan {

// Byte decoding utilities
uint64_t getUInt64LE(const uint8_t* data);
uint32_t getUInt32LE(const uint8_t* data);
uint32_t getUInt16LE(const uint8_t* data);

struct VulkanDelegateHeader {
  bool is_valid() const;

  static executorch::runtime::Result<VulkanDelegateHeader> parse(
      const void* data);

  uint32_t header_size;
  uint32_t flatbuffer_offset;
  uint32_t flatbuffer_size;
  uint32_t bytes_offset;
  uint64_t bytes_size;
};

} // namespace vulkan
} // namespace backends
} // namespace executorch
