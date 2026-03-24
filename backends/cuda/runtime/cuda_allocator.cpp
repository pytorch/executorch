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

Result<void*> CudaAllocator::allocate(size_t nbytes, DeviceIndex index) {
  void* ptr = nullptr;
  cudaError_t prev_device_err = cudaSuccess;
  int prev_device = 0;

  if (index >= 0) {
    prev_device_err = cudaGetDevice(&prev_device);
    if (prev_device_err == cudaSuccess) {
      cudaSetDevice(index);
    }
  }

  cudaError_t err = cudaMalloc(&ptr, nbytes);

  if (index >= 0 && prev_device_err == cudaSuccess) {
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
