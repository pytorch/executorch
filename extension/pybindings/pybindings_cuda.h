/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef EXECUTORCH_PYBINDINGS_CUDA
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/extension/cuda/device_guard.h>
#include <executorch/extension/cuda/runtime_api.h>
#endif

namespace executorch::extension::pybindings {

inline std::shared_ptr<void> allocate_cuda_result(size_t nbytes, int index) {
#ifdef EXECUTORCH_PYBINDINGS_CUDA
  const auto type = executorch::runtime::etensor::DeviceType::CUDA;
  auto* allocator = executorch::runtime::get_device_allocator(type);
  if (allocator == nullptr) {
    // Windows ships the allocator without registering the CUDA delegate.
    allocator = &executorch::extension::cuda::CudaAllocator::instance();
  }
  auto allocation = allocator->allocate(nbytes, index);
  if (!allocation.ok()) {
    throw std::runtime_error(
        "Failed to allocate CUDA memory for an ExecuTorchResult");
  }
  return std::shared_ptr<void>(
      allocation.get(),
      [allocator, index](void* data) { allocator->deallocate(data, index); });
#else
  (void)nbytes;
  (void)index;
  throw std::runtime_error(
      "CUDA tensor interchange requires bindings built with CUDA support");
#endif
}

// A tensor handed across the Python boundary carries no readiness event, and
// another library may write it on any stream, so the boundary waits for the
// whole device, even inside an executorch.runtime.cuda stream block.
inline void synchronize_cuda_device(int index) {
#ifdef EXECUTORCH_PYBINDINGS_CUDA
  auto guard = executorch::extension::cuda::CUDAGuard::create(index);
  if (!guard.ok()) {
    throw std::runtime_error(
        "Failed to select CUDA device " + std::to_string(index));
  }
  const auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("Failed to synchronize CUDA tensor storage: ") +
        cudaGetErrorString(status));
  }
#else
  (void)index;
  throw std::runtime_error(
      "CUDA tensor interchange requires bindings built with CUDA support");
#endif
}

// Checked before selecting the device: a failed selection leaves an error
// pending that the next CUDA call, even in another library, would report.
inline void require_cuda_device(int index) {
#ifdef EXECUTORCH_PYBINDINGS_CUDA
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error(
        std::string("Failed to query CUDA devices: ") +
        cudaGetErrorString(status));
  }
  if (index < 0 || index >= count) {
    throw std::invalid_argument(
        "CUDA device " + std::to_string(index) + " is not available; " +
        std::to_string(count) + " CUDA devices are visible");
  }
#else
  (void)index;
  throw std::runtime_error(
      "CUDA tensor interchange requires bindings built with CUDA support");
#endif
}

// Values match cudaMemcpyKind, so a direction converts to it directly.
enum class CudaCopy {
  HostToDevice = 1,
  DeviceToHost = 2,
  DeviceToDevice = 3,
};

// Runs `function` with CUDA device `index` current, then restores the previous
// device. Some DLPack producers export a CUDA tensor only while its device is
// the current one.
template <typename Function>
auto with_cuda_device(int index, Function&& function) {
#ifdef EXECUTORCH_PYBINDINGS_CUDA
  require_cuda_device(index);
  auto guard = executorch::extension::cuda::CUDAGuard::create(index);
  if (!guard.ok()) {
    throw std::runtime_error(
        "Failed to select CUDA device " + std::to_string(index));
  }
  return function();
#else
  // Without CUDA support, using the capsule fails later with a clear error.
  (void)index;
  return function();
#endif
}

// Copies on the legacy default stream. A copy to the host returns once it has
// finished. A copy to the device may still be in flight when this returns, but
// it runs before anything queued afterwards on a blocking stream, including the
// delegates' cudaStreamPerThread; a result holding such memory still waits for
// the device before another library or a call reads it.
inline void copy_cuda_storage(
    void* destination,
    const void* source,
    size_t nbytes,
    int index,
    CudaCopy direction = CudaCopy::DeviceToDevice) {
#ifdef EXECUTORCH_PYBINDINGS_CUDA
  auto guard = executorch::extension::cuda::CUDAGuard::create(index);
  if (!guard.ok()) {
    throw std::runtime_error(
        "Failed to select CUDA device " + std::to_string(index));
  }
  const auto status = cudaMemcpy(
      destination, source, nbytes, static_cast<cudaMemcpyKind>(direction));
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("Failed to copy CUDA tensor storage: ") +
        cudaGetErrorString(status));
  }
#else
  (void)destination;
  (void)source;
  (void)nbytes;
  (void)index;
  (void)direction;
  throw std::runtime_error(
      "CUDA tensor interchange requires bindings built with CUDA support");
#endif
}

} // namespace executorch::extension::pybindings
