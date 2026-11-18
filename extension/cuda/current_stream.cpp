/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/cuda/current_stream.h>

#include <unordered_map>

#if !defined(EXECUTORCH_USE_HIP)
#include <cuda_runtime.h>
#endif

namespace executorch::extension::cuda::internal {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {

thread_local std::unordered_map<int8_t, cudaStream_t> current_streams_;

} // namespace

Error setCurrentCUDAStream(cudaStream_t stream, int8_t device_index) {
  if (device_index == -1) {
    // CUDA reports devices as int; the index is int8_t to match ATen.
    int tmp_device = -1;
    cudaError_t err = cudaGetDevice(&tmp_device);
    ET_CHECK_OR_RETURN_ERROR(
        err == cudaSuccess,
        Internal,
        "setCurrentCUDAStream: cudaGetDevice failed: %s",
        cudaGetErrorString(err));
    device_index = static_cast<int8_t>(tmp_device);
  }

  current_streams_[device_index] = stream;
  return Error::Ok;
}

Result<cudaStream_t> getCurrentCUDAStream(int8_t device_index) {
  if (device_index == -1) {
    int tmp_device = -1;
    cudaError_t err = cudaGetDevice(&tmp_device);
    ET_CHECK_OR_RETURN_ERROR(
        err == cudaSuccess,
        Internal,
        "getCurrentCUDAStream: cudaGetDevice failed: %s",
        cudaGetErrorString(err));
    device_index = static_cast<int8_t>(tmp_device);
  }

  auto it = current_streams_.find(device_index);
  if (it != current_streams_.end()) {
    return it->second;
  }
  return cudaStreamPerThread;
}

std::optional<cudaStream_t> peekCurrentCUDAStream(int8_t device_index) {
  if (device_index == -1) {
    int tmp_device = -1;
    if (cudaGetDevice(&tmp_device) != cudaSuccess) {
      return std::nullopt;
    }
    device_index = static_cast<int8_t>(tmp_device);
  }

  auto it = current_streams_.find(device_index);
  if (it == current_streams_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void clearCurrentCUDAStream(int8_t device_index) {
  if (device_index == -1) {
    int tmp_device = -1;
    if (cudaGetDevice(&tmp_device) != cudaSuccess) {
      return;
    }
    device_index = static_cast<int8_t>(tmp_device);
  }
  current_streams_.erase(device_index);
}

} // namespace executorch::extension::cuda::internal
