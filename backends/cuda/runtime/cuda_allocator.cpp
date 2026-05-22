/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_allocator.h>

#include <cuda_runtime.h>

#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

Result<void*>
CudaAllocator::allocate(size_t nbytes, DeviceIndex index, size_t alignment) {
  // index == -1 means "use the current CUDA device"; any value < -1 is invalid.
  ET_CHECK_OR_RETURN_ERROR(
      index >= -1,
      InvalidArgument,
      "CudaAllocator::allocate: invalid device index %d (must be >= -1)",
      static_cast<int>(index));

  // Alignment must be a non-zero power of 2.
  ET_CHECK_OR_RETURN_ERROR(
      alignment != 0 && (alignment & (alignment - 1)) == 0,
      InvalidArgument,
      "CudaAllocator::allocate: alignment must be a power of 2, got %zu",
      alignment);

  // cudaMalloc is documented to return memory aligned to at least 256 bytes,
  // which trivially satisfies kDefaultAlignment (alignof(void*)). For any
  // requested alignment <= 256 bytes, the returned pointer is already aligned.
  // Stricter alignment would require over-allocation plus bookkeeping that
  // deallocate() does not currently support, so reject that case.
  constexpr size_t kCudaMallocAlignment = 256;
  ET_CHECK_OR_RETURN_ERROR(
      alignment <= kCudaMallocAlignment,
      NotSupported,
      "CudaAllocator::allocate: requested alignment %zu exceeds cudaMalloc's "
      "guaranteed alignment of %zu bytes; stricter alignment is not supported",
      alignment,
      kCudaMallocAlignment);

  void* ptr = nullptr;
  int prev_device = 0;
  cudaError_t prev_device_err = cudaGetDevice(&prev_device);

  // If index == -1, fall back to the current device returned by cudaGetDevice
  // and skip the set/restore round-trip.
  const bool switch_device = index >= 0 && prev_device_err == cudaSuccess &&
      static_cast<int>(index) != prev_device;
  if (switch_device) {
    cudaSetDevice(index);
  }

  cudaError_t err = cudaMalloc(&ptr, nbytes);

  if (switch_device) {
    cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMalloc failed: %s (requested %zu bytes on device %d)",
        cudaGetErrorString(err),
        nbytes,
        static_cast<int>(index));
    return Error::MemoryAllocationFailed;
  }

  // Sanity check: the pointer returned by cudaMalloc should already meet the
  // requested alignment. If a future CUDA runtime weakens this guarantee, we
  // want to fail loudly rather than silently return a misaligned pointer.
  if ((reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) != 0) {
    ET_LOG(
        Error,
        "cudaMalloc returned pointer %p not aligned to %zu bytes",
        ptr,
        alignment);
    cudaFree(ptr);
    return Error::MemoryAllocationFailed;
  }

  return ptr;
}

void CudaAllocator::deallocate(void* ptr, DeviceIndex index) {
  if (ptr == nullptr) {
    return;
  }

  int prev_device = 0;
  cudaError_t prev_device_err = cudaSuccess;

  if (index >= 0) {
    prev_device_err = cudaGetDevice(&prev_device);
    if (prev_device_err == cudaSuccess) {
      cudaSetDevice(index);
    }
  }

  cudaError_t err = cudaFree(ptr);

  if (index >= 0 && prev_device_err == cudaSuccess) {
    cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaFree failed: %s (ptr=%p, device %d)",
        cudaGetErrorString(err),
        ptr,
        static_cast<int>(index));
  }
}

// TODO(gasoonjia): Add support for async copy
Error CudaAllocator::copy_host_to_device(
    void* dst,
    const void* src,
    size_t nbytes,
    DeviceIndex index) {
  int prev_device = 0;
  cudaError_t prev_device_err = cudaSuccess;

  if (index >= 0) {
    prev_device_err = cudaGetDevice(&prev_device);
    if (prev_device_err == cudaSuccess) {
      cudaSetDevice(index);
    }
  }

  cudaError_t err = cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);

  if (index >= 0 && prev_device_err == cudaSuccess) {
    cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMemcpy H2D failed: %s (%zu bytes, device %d)",
        cudaGetErrorString(err),
        nbytes,
        static_cast<int>(index));
    return Error::Internal;
  }
  return Error::Ok;
}

// TODO(gasoonjia): Add support for async copy
Error CudaAllocator::copy_device_to_host(
    void* dst,
    const void* src,
    size_t nbytes,
    DeviceIndex index) {
  int prev_device = 0;
  cudaError_t prev_device_err = cudaSuccess;

  if (index >= 0) {
    prev_device_err = cudaGetDevice(&prev_device);
    if (prev_device_err == cudaSuccess) {
      cudaSetDevice(index);
    }
  }

  cudaError_t err = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);

  if (index >= 0 && prev_device_err == cudaSuccess) {
    cudaSetDevice(prev_device);
  }

  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMemcpy D2H failed: %s (%zu bytes, device %d)",
        cudaGetErrorString(err),
        nbytes,
        static_cast<int>(index));
    return Error::Internal;
  }
  return Error::Ok;
}

DeviceType CudaAllocator::device_type() const {
  return DeviceType::CUDA;
}

CudaAllocator& CudaAllocator::instance() {
  static CudaAllocator allocator;
  return allocator;
}

Result<void*> CudaAllocator::allocate_async(
    size_t nbytes,
    DeviceIndex index,
    cudaStream_t stream) {
  void* ptr = nullptr;
  cudaError_t err = cudaMallocAsync(&ptr, nbytes, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMallocAsync failed: %s (requested %zu bytes on device %d)",
        cudaGetErrorString(err),
        nbytes,
        static_cast<int>(index));
    return Error::MemoryAllocationFailed;
  }
  return ptr;
}

void CudaAllocator::deallocate_async(
    void* ptr,
    DeviceIndex index,
    cudaStream_t stream) {
  if (ptr == nullptr) {
    return;
  }
  cudaError_t err = cudaFreeAsync(ptr, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaFreeAsync failed: %s (ptr=%p, device %d)",
        cudaGetErrorString(err),
        ptr,
        static_cast<int>(index));
  }
}

Error CudaAllocator::memcpy_async(
    void* dst,
    const void* src,
    size_t nbytes,
    cudaMemcpyKind direction,
    cudaStream_t stream) {
  cudaError_t err = cudaMemcpyAsync(dst, src, nbytes, direction, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "cudaMemcpyAsync failed: %s (%zu bytes)",
        cudaGetErrorString(err),
        nbytes);
    return Error::Internal;
  }
  return Error::Ok;
}

} // namespace executorch::backends::cuda
