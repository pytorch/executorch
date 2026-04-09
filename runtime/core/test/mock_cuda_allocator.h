/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include <cstring>

#include <executorch/runtime/core/device_allocator.h>

namespace executorch {
namespace runtime {
namespace testing {

/**
 * Mock CUDA allocator for testing device memory workflows.
 *
 * Uses host memory (malloc/free/memcpy) to simulate device memory operations,
 * enabling end-to-end data roundtrip verification without requiring actual
 * CUDA hardware. Tracks all allocate/deallocate/copy calls with counters
 * and argument capture for lifecycle verification.
 */
class MockCudaAllocator : public DeviceAllocator {
 public:
  Result<void*> allocate(size_t nbytes, etensor::DeviceIndex index) override {
    void* ptr = std::malloc(nbytes);
    if (!ptr) {
      return Error::MemoryAllocationFailed;
    }
    allocate_count_++;
    last_allocate_size_ = nbytes;
    last_allocate_index_ = index;
    last_allocate_ptr_ = ptr;
    return ptr;
  }

  void deallocate(void* ptr, etensor::DeviceIndex index) override {
    deallocate_count_++;
    last_deallocate_ptr_ = ptr;
    last_deallocate_index_ = index;
    std::free(ptr);
  }

  Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      etensor::DeviceIndex index) override {
    std::memcpy(dst, src, nbytes);
    h2d_count_++;
    last_h2d_dst_ = dst;
    last_h2d_src_ = src;
    last_h2d_size_ = nbytes;
    last_h2d_index_ = index;
    return Error::Ok;
  }

  Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      etensor::DeviceIndex index) override {
    std::memcpy(dst, src, nbytes);
    d2h_count_++;
    last_d2h_dst_ = dst;
    last_d2h_src_ = src;
    last_d2h_size_ = nbytes;
    last_d2h_index_ = index;
    return Error::Ok;
  }

  etensor::DeviceType device_type() const override {
    return etensor::DeviceType::CUDA;
  }

  /**
   * Returns true if ptr falls within the most recent allocation range.
   * Useful for verifying that tensor data_ptrs point to device memory.
   */
  bool is_device_ptr(const void* ptr) const {
    if (last_allocate_ptr_ == nullptr || last_allocate_size_ == 0) {
      return false;
    }
    auto* p = static_cast<const uint8_t*>(ptr);
    auto* base = static_cast<const uint8_t*>(last_allocate_ptr_);
    return p >= base && p < base + last_allocate_size_;
  }

  void reset() {
    allocate_count_ = 0;
    deallocate_count_ = 0;
    h2d_count_ = 0;
    d2h_count_ = 0;
    last_allocate_size_ = 0;
    last_allocate_index_ = -1;
    last_allocate_ptr_ = nullptr;
    last_deallocate_ptr_ = nullptr;
    last_deallocate_index_ = -1;
    last_h2d_dst_ = nullptr;
    last_h2d_src_ = nullptr;
    last_h2d_size_ = 0;
    last_h2d_index_ = -1;
    last_d2h_dst_ = nullptr;
    last_d2h_src_ = nullptr;
    last_d2h_size_ = 0;
    last_d2h_index_ = -1;
  }

  // Allocation tracking
  int allocate_count_ = 0;
  int deallocate_count_ = 0;
  size_t last_allocate_size_ = 0;
  etensor::DeviceIndex last_allocate_index_ = -1;
  void* last_allocate_ptr_ = nullptr;
  void* last_deallocate_ptr_ = nullptr;
  etensor::DeviceIndex last_deallocate_index_ = -1;

  // Host-to-device copy tracking
  int h2d_count_ = 0;
  void* last_h2d_dst_ = nullptr;
  const void* last_h2d_src_ = nullptr;
  size_t last_h2d_size_ = 0;
  etensor::DeviceIndex last_h2d_index_ = -1;

  // Device-to-host copy tracking
  int d2h_count_ = 0;
  void* last_d2h_dst_ = nullptr;
  const void* last_d2h_src_ = nullptr;
  size_t last_d2h_size_ = 0;
  etensor::DeviceIndex last_d2h_index_ = -1;
};

} // namespace testing
} // namespace runtime
} // namespace executorch
