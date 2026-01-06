/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef CUDA_AVAILABLE

#include <cuda.h>
#include <cuda_runtime.h>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>

namespace executorch::backends::aoti::slim::cuda {

/**
 * CUDAGuard - RAII class that sets the current CUDA device.
 *
 * This class saves the current CUDA device on construction and restores it
 * on destruction, providing exception-safe device switching.
 *
 * Thread Safety: NOT THREAD-SAFE
 * - Must only be used within a single thread
 */
struct CUDAGuard {
  /// No default constructor - device must be specified.
  CUDAGuard() = delete;

  /// Sets the current CUDA device to the specified device index.
  /// @param device_index The CUDA device index to switch to.
  explicit CUDAGuard(c10::DeviceIndex device_index) {
    set_index(device_index);
  }

  /// Sets the current CUDA device to the specified device.
  /// @param device The CUDA device to switch to. Must be a CUDA device.
  explicit CUDAGuard(c10::Device device) {
    ET_CHECK_MSG(device.is_cuda(), "Expected a CUDA device for CUDAGuard");
    set_index(device.index());
  }

  // Copy is not allowed
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // Move is not allowed
  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;

  /// Restores the original CUDA device on destruction.
  ~CUDAGuard() {
    if (original_device_index_ != current_device_index_) {
      ET_CUDA_LOG_WARN(cudaSetDevice(original_device_index_));
    }
  }

  /// Sets the CUDA device to the given device index.
  /// @param device_index The device index to switch to.
  void set_index(c10::DeviceIndex device_index) {
    int orig_index = -1;
    ET_CUDA_CHECK(cudaGetDevice(&orig_index));

    original_device_index_ = orig_index;
    current_device_index_ = device_index;
    if (current_device_index_ != original_device_index_) {
      ET_CUDA_CHECK(cudaSetDevice(current_device_index_));
    }
  }

 private:
  c10::DeviceIndex original_device_index_;
  c10::DeviceIndex current_device_index_;
};

} // namespace executorch::backends::aoti::slim::cuda

#endif // CUDA_AVAILABLE
