/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/device_allocator.h>

#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {

DeviceAllocatorRegistry& DeviceAllocatorRegistry::instance() {
  static DeviceAllocatorRegistry registry;
  return registry;
}

void DeviceAllocatorRegistry::register_allocator(
    etensor::DeviceType type,
    DeviceAllocator* alloc) {
  auto index = static_cast<size_t>(type);
  ET_CHECK_MSG(
      index < etensor::kNumDeviceTypes,
      "Invalid device type: %d",
      static_cast<int>(type));
  ET_CHECK_MSG(
      allocators_[index] == nullptr,
      "Allocator already registered for device type: %d",
      static_cast<int>(type));
  allocators_[index] = alloc;
}

DeviceAllocator* DeviceAllocatorRegistry::get_allocator(
    etensor::DeviceType type) {
  auto index = static_cast<size_t>(type);
  if (index >= etensor::kNumDeviceTypes) {
    return nullptr;
  }
  return allocators_[index];
}

// Convenience free functions

void register_device_allocator(
    etensor::DeviceType type,
    DeviceAllocator* alloc) {
  DeviceAllocatorRegistry::instance().register_allocator(type, alloc);
}

DeviceAllocator* get_device_allocator(etensor::DeviceType type) {
  return DeviceAllocatorRegistry::instance().get_allocator(type);
}

} // namespace runtime
} // namespace executorch
