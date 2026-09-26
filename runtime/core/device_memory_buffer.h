/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch::runtime {

/**
 * RAII wrapper that owns a single device memory allocation.
 *
 * On destruction, calls DeviceAllocator::deallocate() to free the memory.
 * This mirrors the role of std::vector<uint8_t> for CPU planned buffers,
 * but for device memory (CUDA, etc.).
 *
 * Move-only: cannot be copied, but can be moved to transfer ownership.
 */
class DeviceMemoryBuffer final {
 public:
  /**
   * Creates a DeviceMemoryBuffer by allocating device memory.
   *
   * Looks up the DeviceAllocator for the given device type via the
   * DeviceAllocatorRegistry. If no allocator is registered for the type,
   * returns Error::NotFound.
   *
   * @param size Number of bytes to allocate.
   * @param type The device type (e.g., CUDA).
   * @param index The device index (e.g., 0 for cuda:0).
   * @param alignment Minimum alignment of the returned pointer in bytes.
   *     Must be a power of 2. Defaults to DeviceAllocator::kDefaultAlignment.
   * @return A Result containing the DeviceMemoryBuffer on success, or an error.
   */
  static Result<DeviceMemoryBuffer> create(
      size_t size,
      etensor::DeviceType type,
      etensor::DeviceIndex index = 0,
      size_t alignment = DeviceAllocator::kDefaultAlignment);

  DeviceMemoryBuffer() = default;

  ~DeviceMemoryBuffer() {
    if (ptr_ != nullptr && allocator_ != nullptr) {
      allocator_->deallocate(ptr_, device_index_);
    }
  }

  // Move constructor: transfer ownership.
  DeviceMemoryBuffer(DeviceMemoryBuffer&& other) noexcept
      : ptr_(other.ptr_),
        size_(other.size_),
        allocator_(other.allocator_),
        device_index_(other.device_index_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.allocator_ = nullptr;
  }

  // Move assignment: release current, take ownership.
  DeviceMemoryBuffer& operator=(DeviceMemoryBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr && allocator_ != nullptr) {
        allocator_->deallocate(ptr_, device_index_);
      }
      ptr_ = other.ptr_;
      size_ = other.size_;
      allocator_ = other.allocator_;
      device_index_ = other.device_index_;
      other.ptr_ = nullptr;
      other.size_ = 0;
      other.allocator_ = nullptr;
    }
    return *this;
  }

  // Non-copyable.
  DeviceMemoryBuffer(const DeviceMemoryBuffer&) = delete;
  DeviceMemoryBuffer& operator=(const DeviceMemoryBuffer&) = delete;

  /// Returns the device pointer, or nullptr if empty/moved-from.
  void* data() const {
    return ptr_;
  }

  /// Returns the size in bytes of the allocation.
  size_t size() const {
    return size_;
  }

  /**
   * Returns a Span<uint8_t> wrapping the device pointer.
   *
   * This is intended for use with HierarchicalAllocator, which only performs
   * pointer arithmetic on the span data and never dereferences it. Device
   * pointers are valid for pointer arithmetic from the CPU side.
   */
  Span<uint8_t> as_span() const {
    return {static_cast<uint8_t*>(ptr_), size_};
  }

 private:
  DeviceMemoryBuffer(
      void* ptr,
      size_t size,
      DeviceAllocator* allocator,
      etensor::DeviceIndex device_index)
      : ptr_(ptr),
        size_(size),
        allocator_(allocator),
        device_index_(device_index) {}

  void* ptr_ = nullptr;
  size_t size_ = 0;
  DeviceAllocator* allocator_ = nullptr;
  etensor::DeviceIndex device_index_ = 0;
};

} // namespace executorch::runtime
