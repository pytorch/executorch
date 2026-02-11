/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <memory>

namespace executorch {
namespace backends {
namespace cuda {

// Shared CUDA stream wrapper with proper RAII cleanup.
// This ensures the stream is destroyed when all handles using it are destroyed.
struct CudaStreamDeleter {
  void operator()(cudaStream_t* stream) const {
    if (stream != nullptr && *stream != nullptr) {
      cudaStreamDestroy(*stream);
    }
    delete stream;
  }
};

// Creates a new shared CUDA stream.
// Returns nullptr on failure.
inline std::shared_ptr<cudaStream_t> create_cuda_stream() {
  cudaStream_t stream;
  cudaError_t err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return std::shared_ptr<cudaStream_t>(
      new cudaStream_t(stream), CudaStreamDeleter());
}

// CUDA-specific delegate handle that extends AOTIDelegateHandle.
// This consolidates CUDA stream management into a single location.
struct CudaDelegateHandle : public aoti::AOTIDelegateHandle {
  // CUDA stream for this handle, support both shared mode and single mode.
  // In shared mode, all cuda delegate handles share the same stream (e.g., for
  // skip-copy optimization), they will all hold a reference to the same
  // shared_ptr. The stream is automatically destroyed when the last handle is
  // destroyed. In single mode, every cuda delegate handle has its own stream.
  std::shared_ptr<cudaStream_t> cuda_stream;

  // Get the raw CUDA stream pointer for use in CUDA API calls.
  // Returns nullptr if no stream is set.
  cudaStream_t get_cuda_stream() const {
    return cuda_stream ? *cuda_stream : nullptr;
  }

  // Check if this handle has a valid CUDA stream.
  bool has_cuda_stream() const {
    return cuda_stream != nullptr && *cuda_stream != nullptr;
  }
};

} // namespace cuda
} // namespace backends
} // namespace executorch
