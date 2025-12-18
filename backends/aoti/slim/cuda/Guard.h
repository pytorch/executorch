#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>

namespace executorch::backends::aoti::slim::cuda {

// Thread-local stream management
namespace detail {
inline thread_local std::unordered_map<
    executorch::backends::aoti::slim::c10::DeviceIndex,
    cudaStream_t>
    current_streams_;
}

/// Set the current CUDA stream for the specified device
inline void setCurrentCUDAStream(
    cudaStream_t stream,
    executorch::backends::aoti::slim::c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    // Get current device if not specified
    int current_device;
    STANDALONE_CUDA_CHECK(cudaGetDevice(&current_device));
    device_index = current_device;
  }

  detail::current_streams_[device_index] = stream;
}

/// Get the current CUDA stream for the specified device
inline cudaStream_t getCurrentCUDAStream(
    executorch::backends::aoti::slim::c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    // Get current device if not specified
    int current_device;
    STANDALONE_CUDA_CHECK(cudaGetDevice(&current_device));
    device_index = current_device;
  }

  auto it = detail::current_streams_.find(device_index);
  if (it != detail::current_streams_.end()) {
    return it->second;
  }

  // Create a new stream and set it as current
  cudaStream_t stream;
  STANDALONE_CUDA_CHECK(cudaStreamCreate(&stream));
  setCurrentCUDAStream(stream, device_index);
  return stream;
}

struct CUDAGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit CUDAGuard() = delete;

  /// Set the current CUDA device to the passed device index.
  explicit CUDAGuard(
      executorch::backends::aoti::slim::c10::DeviceIndex device_index) {
    set_index(device_index);
  }

  /// Sets the current CUDA device to the passed device.  Errors if the passed
  /// device is not a CUDA device.
  explicit CUDAGuard(executorch::backends::aoti::slim::c10::Device device) {
    STANDALONE_CHECK(
        device.is_cuda(),
        "Expected a CUDA device for CUDAGuard, but got ",
        device);
    set_index(device.index());
  }

  // Copy is not allowed
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;

  ~CUDAGuard() {
    // Restore the original device if necessary
    if (original_device_index_ != current_device_index_) {
      STANDALONE_CUDA_CHECK_WARN(cudaSetDevice(original_device_index_));
    }
  }

  /// Sets the CUDA device to the given device index.
  void set_index(
      executorch::backends::aoti::slim::c10::DeviceIndex device_index) {
    int orig_index = -1;
    STANDALONE_CUDA_CHECK(cudaGetDevice(&orig_index));

    original_device_index_ = orig_index;
    current_device_index_ = device_index;
    if (current_device_index_ != original_device_index_) {
      STANDALONE_CUDA_CHECK(cudaSetDevice(current_device_index_));
    }
  }

 private:
  /// The guard for the current device.
  executorch::backends::aoti::slim::c10::DeviceIndex original_device_index_;
  executorch::backends::aoti::slim::c10::DeviceIndex current_device_index_;
};

struct CUDAStreamGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit CUDAStreamGuard() = delete;

  /// Set the current CUDA stream to the passed stream on the specified device.
  explicit CUDAStreamGuard(
      cudaStream_t stream,
      executorch::backends::aoti::slim::c10::DeviceIndex device_index)
      : device_guard_(device_index) {
    set_stream(stream, device_index);
  }

  // Copy is not allowed
  CUDAStreamGuard(const CUDAStreamGuard&) = delete;
  CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  CUDAStreamGuard(CUDAStreamGuard&& other) = delete;
  CUDAStreamGuard& operator=(CUDAStreamGuard&& other) = delete;

  ~CUDAStreamGuard() {
    // Restore the original stream for the device
    setCurrentCUDAStream(original_stream_, device_index_);
    // Device guard will automatically restore the original device
  }

  /// Sets the CUDA stream to the given stream on the specified device.
  void set_stream(
      cudaStream_t stream,
      executorch::backends::aoti::slim::c10::DeviceIndex device_index) {
    // Store the original stream for this device
    original_stream_ = getCurrentCUDAStream(device_index);
    current_stream_ = stream;
    device_index_ = device_index;

    // Set the new stream as current for this device
    setCurrentCUDAStream(stream, device_index);
  }

  /// Get the current guarded stream
  cudaStream_t stream() const {
    return current_stream_;
  }

  /// Get the device index being guarded
  executorch::backends::aoti::slim::c10::DeviceIndex device_index() const {
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
  executorch::backends::aoti::slim::c10::DeviceIndex device_index_;
};

} // namespace executorch::backends::aoti::slim::cuda
