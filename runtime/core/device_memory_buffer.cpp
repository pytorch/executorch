/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/device_memory_buffer.h>

namespace executorch::runtime {

Result<DeviceMemoryBuffer> DeviceMemoryBuffer::create(
    size_t size,
    etensor::DeviceType type,
    etensor::DeviceIndex index,
    size_t alignment) {
  DeviceAllocator* allocator = get_device_allocator(type);
  if (allocator == nullptr) {
    ET_LOG(
        Error,
        "No device allocator registered for device type %d",
        static_cast<int>(type));
    return Error::NotFound;
  }

  auto result = allocator->allocate(size, index, alignment);
  if (!result.ok()) {
    return result.error();
  }

  return DeviceMemoryBuffer(result.get(), size, allocator, index);
}

} // namespace executorch::runtime
