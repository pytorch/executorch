/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <cstdint>

namespace executorch::backends::cuda {

using executorch::runtime::Error;
using executorch::runtime::Result;

// Type alias for device index
using DeviceIndex = int32_t;

/**
 * Set the current CUDA stream for the specified device.
 *
 * @param stream The CUDA stream to set as current
 * @param device_index The device index (-1 to use current device)
 * @return Error code indicating success or failure
 */
Error setCurrentCUDAStream(cudaStream_t stream, DeviceIndex device_index = -1);

/**
 * Get the current CUDA stream for the specified device.
 * If no stream has been set, creates a new stream and sets it as current.
 *
 * @param device_index The device index (-1 to use current device)
 * @return Result containing the current stream on success, or an error code on
 * failure
 */
Result<cudaStream_t> getCurrentCUDAStream(DeviceIndex device_index = -1);

/**
 * RAII guard that sets the current CUDA device and restores it on destruction.
 * This ensures that the device is properly restored even if an exception
 * occurs.
 *
 */
class CUDAGuard {
 private:
  /**
   * Private constructor - use create() factory method instead.
   */
  explicit CUDAGuard()
      : original_device_index_(-1), current_device_index_(-1) {}

 public:
  /**
   * Factory method to create a CUDAGuard.
   *
   * @param device_index The device index to set as current
   * @return Result containing the guard on success, or an error code on failure
   */
  static Result<CUDAGuard> create(DeviceIndex device_index);

  // Copy is not allowed
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // Move constructor and assignment
  CUDAGuard(CUDAGuard&& other) noexcept;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;

  /**
   * Destructor that restores the original device if necessary.
   */
  ~CUDAGuard();

  /**
   * Sets the CUDA device to the given device index.
   *
   * @param device_index The device index to set as current
   * @return Error code indicating success or failure
   */
  Error set_index(DeviceIndex device_index);

  /**
   * Get the original device index before the guard was created.
   *
   * @return The original device index
   */
  DeviceIndex original_device() const {
    return original_device_index_;
  }

  /**
   * Get the current device index.
   *
   * @return The current device index
   */
  DeviceIndex current_device() const {
    return current_device_index_;
  }

 private:
  /// The original device before this guard was created
  DeviceIndex original_device_index_;
  /// The current device managed by this guard
  DeviceIndex current_device_index_;
};

/**
 * RAII guard that sets the current CUDA device and stream, restoring both on
 * destruction. This is useful for temporarily switching to a different device
 * and stream.
 *
 */
class CUDAStreamGuard {
 private:
  // Private constructor that takes a CUDAGuard
  explicit CUDAStreamGuard(CUDAGuard&& guard)
      : device_guard_(std::move(guard)),
        original_stream_(nullptr),
        current_stream_(nullptr),
        device_index_(-1) {}

 public:
  /**
   * Factory method to create a CUDAStreamGuard.
   *
   * @param stream The CUDA stream to set as current
   * @param device_index The device index for the stream
   * @return Result containing the guard on success, or an error code on failure
   */
  static Result<CUDAStreamGuard> create(
      cudaStream_t stream,
      DeviceIndex device_index);

  // Copy is not allowed
  CUDAStreamGuard(const CUDAStreamGuard&) = delete;
  CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;

  // Move constructor and assignment
  CUDAStreamGuard(CUDAStreamGuard&& other) noexcept;
  CUDAStreamGuard& operator=(CUDAStreamGuard&& other) noexcept = delete;

  /**
   * Destructor that restores the original stream and device.
   */
  ~CUDAStreamGuard();

  /**
   * Sets the CUDA stream to the given stream on the specified device.
   *
   * @param stream The CUDA stream to set as current
   * @param device_index The device index for the stream
   * @return Error code indicating success or failure
   */
  Error set_stream(cudaStream_t stream, DeviceIndex device_index);

  /**
   * Get the current guarded stream.
   *
   * @return The current stream
   */
  cudaStream_t stream() const {
    return current_stream_;
  }

  /**
   * Get the device index being guarded.
   *
   * @return The device index
   */
  DeviceIndex device_index() const {
    return device_index_;
  }

 private:
  /// The device guard that handles device switching
  CUDAGuard device_guard_;
  /// The original stream that was current before this guard
  cudaStream_t original_stream_ = nullptr;
  /// The current stream being guarded
  cudaStream_t current_stream_ = nullptr;
  /// The device index for this stream guard
  DeviceIndex device_index_;
};

} // namespace executorch::backends::cuda
