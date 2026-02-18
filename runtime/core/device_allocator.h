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

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/portable_type/device.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace runtime {

/**
 * Abstract interface for device-specific memory allocation.
 *
 * Each device type (CUDA, etc.) provides a concrete implementation
 * that handles memory allocation on that device. Implementations are
 * expected to be singletons with static lifetime, registered via
 * DeviceAllocatorRegistry.

 */
class DeviceAllocator {
 public:
  virtual ~DeviceAllocator() = default;

  /**
   * Initialize a memory buffer pool for memory-planned tensors.
   *
   * @param memory_id The ID of the memory buffer (index into
   *     ExecutionPlan.non_const_buffer_sizes).
   * @param size The size in bytes to allocate for this buffer.
   * @param index The device index (e.g., GPU 0 vs GPU 1).
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error
  init_buffer(uint32_t memory_id, size_t size, etensor::DeviceIndex index) = 0;

  /**
   * Get a pointer to a specific offset within a pre-allocated buffer pool.
   *
   * @param memory_id The ID of the memory buffer.
   * @param offset_bytes Offset in bytes from the start of the buffer.
   * @param size_bytes Size of the requested region in bytes.
   * @param index The device index.
   * @return A Result containing the device pointer on success, or an error.
   */
  virtual Result<void*> get_offset_address(
      uint32_t memory_id,
      size_t offset_bytes,
      size_t size_bytes,
      etensor::DeviceIndex index) = 0;

  /**
   * Allocate device memory.
   *
   * @param nbytes Number of bytes to allocate.
   * @param index The device index.
   * @return A Result containing the device pointer on success, or an error.
   */
  virtual Result<void*> allocate(size_t nbytes, etensor::DeviceIndex index) = 0;

  /**
   * Deallocate device memory previously allocated via allocate().
   *
   * @param ptr Pointer to the memory to deallocate.
   * @param index The device index.
   */
  virtual void deallocate(void* ptr, etensor::DeviceIndex index) = 0;

  /**
   * Copy data from host memory to device memory.
   *
   * @param dst Destination pointer (device memory).
   * @param src Source pointer (host memory).
   * @param nbytes Number of bytes to copy.
   * @param index The device index.
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      etensor::DeviceIndex index) = 0;

  /**
   * Copy data from device memory to host memory.
   *
   * @param dst Destination pointer (host memory).
   * @param src Source pointer (device memory).
   * @param nbytes Number of bytes to copy.
   * @param index The device index.
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      etensor::DeviceIndex index) = 0;

  /**
   * Returns the device type this allocator handles.
   */
  virtual etensor::DeviceType device_type() const = 0;
};

/**
 * Registry for device allocators.
 *
 * Provides a global mapping from DeviceType to DeviceAllocator instances.
 * Device allocators register themselves at static initialization time,
 * and the runtime queries the registry to find the appropriate allocator
 * for a given device type.
 */
class DeviceAllocatorRegistry {
 public:
  /**
   * Returns the singleton instance of the registry.
   */
  static DeviceAllocatorRegistry& instance();

  /**
   * Register an allocator for a specific device type.
   *
   * @param type The device type this allocator handles.
   * @param alloc Pointer to the allocator (must have static lifetime).
   */
  void register_allocator(etensor::DeviceType type, DeviceAllocator* alloc);

  /**
   * Get the allocator for a specific device type.
   *
   * @param type The device type.
   * @return Pointer to the allocator, or nullptr if not registered.
   */
  DeviceAllocator* get_allocator(etensor::DeviceType type);

 private:
  DeviceAllocatorRegistry() = default;

  // Fixed-size array indexed by device type. This avoids dynamic allocation
  // and is suitable for embedded environments.
  DeviceAllocator* allocators_[etensor::kNumDeviceTypes] = {};
};

// Convenience free functions

/**
 * Register a device allocator for a specific device type.
 *
 * @param type The device type this allocator handles.
 * @param alloc Pointer to the allocator (must have static lifetime).
 */
void register_device_allocator(
    etensor::DeviceType type,
    DeviceAllocator* alloc);

/**
 * Get the device allocator for a specific device type.
 *
 * @param type The device type.
 * @return Pointer to the allocator, or nullptr if not registered.
 */
DeviceAllocator* get_device_allocator(etensor::DeviceType type);

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::DeviceAllocator;
using ::executorch::runtime::DeviceAllocatorRegistry;
using ::executorch::runtime::get_device_allocator;
using ::executorch::runtime::register_device_allocator;
} // namespace executor
} // namespace torch
