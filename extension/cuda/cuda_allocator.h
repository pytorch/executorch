/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/cuda/export.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/device_allocator.h>

namespace executorch::extension::cuda {

/**
 * CUDA implementation of DeviceAllocator.
 *
 * Uses cudaMalloc/cudaFree for allocation and cudaMemcpy for host-device
 * transfers. The CUDA backend registers instance() with the
 * DeviceAllocatorRegistry when it is linked. Linking only this library does
 * not register it.
 *
 * All CUDA memory operations in the CUDA backend should go through this
 * allocator for consistent memory management.
 */
class EXECUTORCH_EXTENSION_CUDA_API CudaAllocator final
    : public executorch::runtime::DeviceAllocator {
 public:
  /**
   * Allocates device memory aligned to `alignment`, a power of two. Above 256
   * bytes, the one cudaMalloc call reserves alignment minus one extra bytes
   * and the result is rounded up. Free the result with deallocate(), not
   * cudaFree() or deallocate_async(), since it may not be the pointer
   * cudaMalloc returned.
   */
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
   * Allocate device memory asynchronously on the given CUDA stream, from this
   * allocator's pool for the device. On a device without memory pools it falls
   * back to cudaMalloc, which the caller's stream ordering still covers, and
   * refuses while `stream` is capturing a CUDA graph.
   */
  static executorch::runtime::Result<void*> allocate_async(
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index,
      cudaStream_t stream);

  /**
   * Deallocate device memory asynchronously on the given CUDA stream. Memory on
   * a device without memory pools is released with cudaFree instead, except
   * while `stream` is capturing a CUDA graph, when it is kept because the graph
   * may still use it.
   */
  static void deallocate_async(
      void* ptr,
      executorch::runtime::etensor::DeviceIndex index,
      cudaStream_t stream);

  /**
   * Whether a device supports the stream-ordered allocator (memory pools).
   * Data-center GPUs in TCC mode on Windows do not; there cudaMallocAsync and
   * CUDA graph memory are unavailable, and allocations fall back to cudaMalloc.
   * Always true on ROCm.
   *
   * @param index Device to query, or a negative value for the current one.
   */
  static bool memory_pools_supported(
      executorch::runtime::etensor::DeviceIndex index);

  /**
   * Stream-ordered scratch from the device default pool, which is what
   * cudaMallocAsync on `stream` gives, for callers that do not want this
   * allocator's retaining pool. On a device without memory pools it falls back
   * to cudaMalloc, and refuses while `stream` is capturing a CUDA graph, since
   * a synchronous allocation cannot be recorded into one.
   *
   * @param index The device `stream` belongs to.
   */
  static executorch::runtime::Result<void*> allocate_stream_ordered(
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index,
      cudaStream_t stream);

  /**
   * Frees memory from allocate_stream_ordered on the same device and stream.
   * Returns Error::Internal when the free fails (cudaFreeAsync, or cudaFree on
   * a device without memory pools), and Error::NotSupported when such a device
   * keeps the block because `stream` is capturing a CUDA graph.
   */
  static executorch::runtime::Error deallocate_stream_ordered(
      void* ptr,
      executorch::runtime::etensor::DeviceIndex index,
      cudaStream_t stream);

  /**
   * Return unused memory from this allocator's shared pools to the driver.
   *
   * All delegates using this allocator share its pools and retention threshold.
   * Trimming may release any delegate's cached blocks, making its next
   * allocation slower. Live allocations and the device default pool are
   * unaffected. Unused graph memory is trimmed device-wide only on devices
   * this allocator has a pool entry for.
   *
   * Call after device work has finished. This function does not synchronize;
   * only frees already observed by the driver can be released. The CUDA
   * backend calls it when its last handle is destroyed, even if other
   * delegates still use the allocator.
   *
   * Does nothing on ROCm. HIP has equivalents for all of these calls; this
   * repository's CUDA-to-HIP compatibility header does not alias them yet.
   *
   * @param index Device to release on, or a negative value to release every
   *     device this allocator has a pool entry for. This means every device,
   * not the current one, which is what a negative value means elsewhere in this
   *     class: a delegate is often torn down from a thread that is not current
   *     on the device it ran on, so releasing only the current device would
   * leave that memory held.
   */
  static void release_cached_memory(
      executorch::runtime::etensor::DeviceIndex index);

#if !defined(EXECUTORCH_USE_HIP)
  /**
   * The shared memory pool this allocator uses on a device, or nullptr if it
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
   * The caller supplies the copy direction.
   */
  static executorch::runtime::Error memcpy_async(
      void* dst,
      const void* src,
      size_t nbytes,
      cudaMemcpyKind direction,
      cudaStream_t stream);

 private:
  CudaAllocator() = default;
  CudaAllocator(const CudaAllocator&) = delete;
  CudaAllocator& operator=(const CudaAllocator&) = delete;
  CudaAllocator(CudaAllocator&&) = delete;
  CudaAllocator& operator=(CudaAllocator&&) = delete;
};

} // namespace executorch::extension::cuda

namespace executorch::backends::cuda {
// Preserve source compatibility for callers of the former backend header.
using ::executorch::extension::cuda::CudaAllocator;
} // namespace executorch::backends::cuda
