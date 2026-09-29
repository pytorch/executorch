/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/memory_allocator.h>
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
  /**
   * Default alignment of memory returned by allocate(). Reuses
   * MemoryAllocator::kDefaultAlignment so host- and device-side allocations
   * share the same baseline contract. Backends whose underlying device APIs
   * already provide stronger guarantees (e.g. cudaMalloc returns 256-byte
   * aligned pointers) will trivially satisfy this.
   */
  static constexpr size_t kDefaultAlignment =
      MemoryAllocator::kDefaultAlignment;

  virtual ~DeviceAllocator() = default;
  /**
   * Allocate device memory.
   *
   * @param nbytes Number of bytes to allocate.
   * @param index The device index.
   * @param alignment Minimum alignment of the returned pointer in bytes.
   *     Must be a power of 2. Defaults to kDefaultAlignment.
   * @return A Result containing the device pointer on success, or an error.
   */
  virtual Result<void*> allocate(
      size_t nbytes,
      etensor::DeviceIndex index,
      size_t alignment = kDefaultAlignment) = 0;

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
 *
 * Threading contract:
 * - Registration is expected to happen once per device type during static
 *   initialization (single-threaded). The registry itself does not perform
 *   any locking around register_allocator()/get_allocator(), and concurrent
 *   registration is not supported.
 * - After registration, get_allocator() is safe to call concurrently from
 *   multiple threads because the underlying array is never mutated again.
 * - The DeviceAllocator implementation is responsible for its own
 *   thread-safety. When multiple Programs are loaded concurrently and each
 *   needs device memory, the allocator must serialize access to any shared
 *   state internally (similar to how XNNPACK's weight cache guards its
 *   internal state). The registry does not provide any synchronization on
 *   behalf of the allocator.
 */
class DeviceAllocatorRegistry {
 public:
  /**
   * Returns the singleton instance of the registry.
   */
  static DeviceAllocatorRegistry& instance();

  /**
   * Register an allocator. The device type is taken from
   * alloc->device_type(). Each device type may only be registered once;
   * attempting to register a second allocator for the same device type
   * will abort.
   *
   * Not thread-safe. Expected to be called during static initialization.
   *
   * @param alloc Pointer to the allocator (must have static lifetime).
   */
  void register_allocator(DeviceAllocator* alloc);

  /**
   * Get the allocator for a specific device type.
   *
   * Safe to call concurrently with other get_allocator() calls.
   *
   * @param type The device type.
   * @return Pointer to the allocator, or nullptr if not registered.
   */
  DeviceAllocator* get_allocator(etensor::DeviceType type);

 private:
  DeviceAllocatorRegistry() = default;

  // Singletons must not be copied or moved; instance() returns a reference,
  // and silently shallow-copying the registry would lead to confusing bugs
  // where modifications to the copy don't affect the real singleton.
  DeviceAllocatorRegistry(const DeviceAllocatorRegistry&) = delete;
  DeviceAllocatorRegistry& operator=(const DeviceAllocatorRegistry&) = delete;
  DeviceAllocatorRegistry(DeviceAllocatorRegistry&&) = delete;
  DeviceAllocatorRegistry& operator=(DeviceAllocatorRegistry&&) = delete;

  // Fixed-size array indexed by device type. This avoids dynamic allocation
  // and is suitable for embedded environments.
  DeviceAllocator* allocators_[etensor::kNumDeviceTypes] = {};
};

// Convenience free functions

/**
 * Register a device allocator. The device type is taken from
 * alloc->device_type(). See DeviceAllocatorRegistry::register_allocator()
 * for the threading contract.
 *
 * @param alloc Pointer to the allocator (must have static lifetime).
 */
void register_device_allocator(DeviceAllocator* alloc);

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
