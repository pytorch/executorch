/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef EXECUTORCH_PYBINDINGS_CUDA
#include <executorch/extension/cuda/device_guard.h>
#include <executorch/extension/cuda/runtime_api.h>
#endif

namespace executorch::extension::pybindings {

// The Python boundary is synchronous until the runtime exposes readiness
// events for inputs, output allocations, and caller-selected execution streams.
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

inline void copy_cuda_storage(
    void* destination,
    const void* source,
    size_t nbytes,
    int index) {
#ifdef EXECUTORCH_PYBINDINGS_CUDA
  auto guard = executorch::extension::cuda::CUDAGuard::create(index);
  if (!guard.ok()) {
    throw std::runtime_error(
        "Failed to select CUDA device " + std::to_string(index));
  }
  auto status =
      cudaMemcpy(destination, source, nbytes, cudaMemcpyDeviceToDevice);
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("Failed to clone CUDA tensor storage: ") +
        cudaGetErrorString(status));
  }
#else
  (void)destination;
  (void)source;
  (void)nbytes;
  (void)index;
  throw std::runtime_error(
      "CUDA tensor interchange requires bindings built with CUDA support");
#endif
}

} // namespace executorch::extension::pybindings
