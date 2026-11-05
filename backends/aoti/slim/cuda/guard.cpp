/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/runtime/platform/log.h>
#include <limits>

// Keep this file free of static and thread_local state: a Windows build
// without EXECUTORCH_BUILD_SHARED links it twice (see backends/cuda).

namespace executorch::backends::cuda {

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
          static_cast<int>(device_index_));
    }
  }
}

Error CUDAStreamGuard::set_stream(
    cudaStream_t stream,
    DeviceIndex device_index) {
  auto result = getCurrentCUDAStream(device_index);
  if (!result.ok()) {
    ET_LOG(
        Error,
        "Failed to get current stream for device %d",
        static_cast<int>(device_index));
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

} // namespace executorch::backends::cuda
