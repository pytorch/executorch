/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/guard.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace cuda {

namespace {
// Thread-local stream storage (private to this file)
thread_local std::unordered_map<DeviceIndex, cudaStream_t> current_streams_;
} // namespace

Error setCurrentCUDAStream(cudaStream_t stream, DeviceIndex device_index) {
  if (device_index == -1) {
    // Get current device if not specified
    int current_device;
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetDevice(&current_device));
    device_index = current_device;
  }

  current_streams_[device_index] = stream;
  return Error::Ok;
}

Result<cudaStream_t> getCurrentCUDAStream(DeviceIndex device_index) {
  if (device_index == -1) {
    int current_device;
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetDevice(&current_device));
    device_index = current_device;
  }

  auto it = current_streams_.find(device_index);
  if (it != current_streams_.end()) {
    return it->second;
  }

  cudaStream_t stream;
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamCreate(&stream));
  setCurrentCUDAStream(stream, device_index);
  return stream;
}

CUDAGuard::CUDAGuard(CUDAGuard&& other) noexcept
    : original_device_index_(other.original_device_index_),
      current_device_index_(other.current_device_index_) {
  // Mark the moved-from object as "already restored" so its destructor doesn't
  // try to restore the device
  other.original_device_index_ = other.current_device_index_;
}

CUDAGuard::~CUDAGuard() {
  if (original_device_index_ != current_device_index_) {
    cudaError_t err = cudaSetDevice(original_device_index_);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "~CUDAGuard: Failed to restore device to %d: %s",
          original_device_index_,
          cudaGetErrorString(err));
    }
  }
}

Error CUDAGuard::set_index(DeviceIndex device_index) {
  int orig_index = -1;
  ET_CUDA_CHECK_OR_RETURN_ERROR(cudaGetDevice(&orig_index));

  original_device_index_ = orig_index;
  current_device_index_ = device_index;

  if (current_device_index_ != original_device_index_) {
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaSetDevice(current_device_index_));
  }

  return Error::Ok;
}

Result<CUDAGuard> CUDAGuard::create(DeviceIndex device_index) {
  CUDAGuard guard; // Fixed: Removed () to create a variable, not a function
  ET_CHECK_OK_OR_RETURN_ERROR(guard.set_index(device_index));
  return guard;
}

CUDAStreamGuard::CUDAStreamGuard(CUDAStreamGuard&& other) noexcept
    : device_guard_(std::move(other.device_guard_)),
      original_stream_(other.original_stream_),
      current_stream_(other.current_stream_),
      device_index_(other.device_index_) {
  // Mark the moved-from object as "already restored" so its destructor doesn't
  // try to restore the stream
  other.original_stream_ = other.current_stream_;
}

CUDAStreamGuard::~CUDAStreamGuard() {
  // Restore the original stream unless this object was moved-from.
  // After a move, original_stream_ == current_stream_, which indicates
  // the moved-from object should not restore.
  // Note: nullptr is a valid stream value (represents the default stream),
  // so we must restore even if original_stream_ is nullptr.
  if (original_stream_ != current_stream_) {
    Error err = setCurrentCUDAStream(original_stream_, device_index_);
    if (err != Error::Ok) {
      ET_LOG(
          Error,
          "~CUDAStreamGuard: Failed to restore stream for device %d",
          device_index_);
    }
  }
}

Error CUDAStreamGuard::set_stream(
    cudaStream_t stream,
    DeviceIndex device_index) {
  auto result = getCurrentCUDAStream(device_index);
  if (!result.ok()) {
    ET_LOG(Error, "Failed to get current stream for device %d", device_index);
    return result.error();
  }

  original_stream_ = result.get();
  current_stream_ = stream;
  device_index_ = device_index;

  ET_CHECK_OK_OR_RETURN_ERROR(setCurrentCUDAStream(stream, device_index));

  return Error::Ok;
}

Result<CUDAStreamGuard> CUDAStreamGuard::create(
    cudaStream_t stream,
    DeviceIndex device_index) {
  auto guard_result = CUDAGuard::create(device_index);
  ET_CHECK_OK_OR_RETURN_ERROR(guard_result.error());

  CUDAStreamGuard stream_guard(std::move(guard_result.get()));
  ET_CHECK_OK_OR_RETURN_ERROR(stream_guard.set_stream(stream, device_index));

  return stream_guard;
}

} // namespace cuda
} // namespace backends
} // namespace executorch
