/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/result.h>

namespace torch {
namespace executor {
namespace vulkan {

struct VulkanDelegateHeader {
  bool is_valid() const;

  static Result<VulkanDelegateHeader> Parse(const void* data);

  uint32_t header_size;
  uint32_t flatbuffer_offset;
  uint32_t flatbuffer_size;
  uint32_t bytes_offset;
  uint64_t bytes_size;
};

} // namespace vulkan
} // namespace executor
} // namespace torch
