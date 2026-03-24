/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>

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
      executorch::runtime::etensor::DeviceIndex index) override;

  void deallocate(
      void* ptr,
      executorch::runtime::etensor::DeviceIndex index) override;

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
