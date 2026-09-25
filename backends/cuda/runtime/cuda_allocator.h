/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/device_allocator.h>

namespace executorch::backends::cuda {

/**
 * CUDA implementation of DeviceAllocator.
 *
 * Uses cudaMalloc/cudaFree for allocation and cudaMemcpy for host-device
 * transfers. This allocator is automatically registered as a singleton
 * with the DeviceAllocatorRegistry when the CUDA backend library is linked.
 *
 * All CUDA memory operations in the CUDA backend should go through this
 * allocator for consistent memory management.
 */
class CudaAllocator final : public executorch::runtime::DeviceAllocator {
 public:
  executorch::runtime::Result<void*> allocate(
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index,
      size_t alignment = kDefaultAlignment) override;

  void deallocate(void* ptr, executorch::runtime::etensor::DeviceIndex index)
      override;

  executorch::runtime::Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index) override;

  executorch::runtime::Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index) override;

  executorch::runtime::etensor::DeviceType device_type() const override;

  /// Returns the global CudaAllocator singleton.
  static CudaAllocator& instance();

  // --- Async (stream-based) operations for SlimTensor/Storage layer ---

  /**
   * Allocate device memory asynchronously on the given CUDA stream.
   */
  static executorch::runtime::Result<void*> allocate_async(
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index,
      cudaStream_t stream);

  /**
   * Deallocate device memory asynchronously on the given CUDA stream.
   */
  static void deallocate_async(
      void* ptr,
      executorch::runtime::etensor::DeviceIndex index,
      cudaStream_t stream);

  /**
   * Return memory this backend's device pool is holding for reuse back to the
   * driver.
   *
   * The pool keeps freed memory so that repeated allocations do not have to map
   * it again, which is what makes delegate execution cheap, so a long-lived
   * process should call this once its work on the device is finished. Only
   * frees the driver has already observed can be released, so a caller that has
   * not synchronized simply gets less back. Allocations that are still live are
   * unaffected either way.
   *
   * The pool belongs to this backend rather than being the device default pool,
   * so the pool trim never affects memory another user of the async allocator
   * is holding. The graph memory trim is the exception: it is scoped to the
   * device, so it also releases unused graph memory cached by other users in
   * this process.
   *
   * Does nothing on ROCm. HIP has equivalents for all of these calls; this
   * repository's CUDA-to-HIP compatibility header does not alias them yet.
   *
   * @param index Device to release on, or a negative value to release every
   *     device this backend has allocated on. Note that means every device, not
   *     the current one, which is what a negative value means elsewhere in this
   *     class: a delegate is often torn down from a thread that is not current
   * on the device it ran on, so releasing only the current device would leave
   *     that memory held.
   */
  static void release_cached_memory(
      executorch::runtime::etensor::DeviceIndex index);

#if !defined(EXECUTORCH_USE_HIP)
  /**
   * The memory pool this backend allocates from on a device, or nullptr if it
   * has not allocated there or the pool could not be created.
   *
   * Exposed so a test can observe what the pool is holding, which is not
   * visible through the device default pool. No production caller.
   *
   * Not declared on ROCm: the pool code is compiled out there, so there is
   * nothing to observe and the pool type needs no HIP alias.
   *
   * @param index Device to query, or a negative value for the current one.
   */
  static cudaMemPool_t pool_for_device(
      executorch::runtime::etensor::DeviceIndex index);
#endif // !EXECUTORCH_USE_HIP

  /**
   * Copy memory asynchronously on the given CUDA stream.
   * Supports H2D, D2H, and D2D based on src/dst device types.
   */
  static executorch::runtime::Error memcpy_async(
      void* dst,
      const void* src,
      size_t nbytes,
      cudaMemcpyKind direction,
      cudaStream_t stream);
};

} // namespace executorch::backends::cuda
