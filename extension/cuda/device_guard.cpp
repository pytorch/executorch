/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/cuda/device_guard.h>

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>

#if defined(EXECUTORCH_USE_HIP)
#include <executorch/extension/cuda/runtime_api.h>
#else
#include <cuda_runtime.h>
#endif

namespace executorch::extension::cuda {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

Result<CUDAGuard> CUDAGuard::create(int device_index) {
  // Where a build makes no shared runtime, this library carries its own copy of
  // the platform layer, and the log below would be the first thing to touch it.
  // Safe to call more than once, which is why the kernel registry does the same
  // at its own entry point.
  ::et_pal_init();
  CUDAGuard guard;
  ET_CHECK_OK_OR_RETURN_ERROR(guard.set_index(device_index));
  return guard;
}

CUDAGuard::CUDAGuard(CUDAGuard&& other) noexcept
    : original_device_index_(other.original_device_index_),
      current_device_index_(other.current_device_index_) {
  other.original_device_index_ = other.current_device_index_;
}

CUDAGuard::~CUDAGuard() {
  // Keyed on what was asked for, not on whether the selection reported success.
  // A failed cudaSetDevice can still move the calling thread, which is what
  // happens once an error is already pending on the device.
  if (original_device_index_ != current_device_index_) {
    cudaError_t err = cudaSetDevice(original_device_index_);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "~CUDAGuard: failed to restore device to %d: %s",
          original_device_index_,
          cudaGetErrorString(err));
    }
  }
}

Error CUDAGuard::set_index(int device_index) {
  int current = -1;
  cudaError_t err = cudaGetDevice(&current);
  if (err != cudaSuccess) {
    ET_LOG(
        Error, "CUDAGuard: cudaGetDevice failed: %s", cudaGetErrorString(err));
    return Error::Internal;
  }
  original_device_index_ = current;
  current_device_index_ = device_index;
  if (current_device_index_ != original_device_index_) {
    err = cudaSetDevice(current_device_index_);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "CUDAGuard: cudaSetDevice(%d) failed: %s",
          device_index,
          cudaGetErrorString(err));
      return Error::Internal;
    }
  }
  return Error::Ok;
}

} // namespace executorch::extension::cuda
