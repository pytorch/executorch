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
#include <vector>

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

  // --- CUDA graph state ---
  // Phase: 0=disabled, 1=warmup, 2=captured (replay mode)
  int cuda_graph_phase = 0;
  int cuda_graph_warmup_remaining = 0;

  // Captured graph and executable instance
  cudaGraph_t cuda_graph = nullptr;
  cudaGraphExec_t cuda_graph_exec = nullptr;

  // Static input/output GPU buffers pinned during capture.
  // These hold the tensor metadata; the underlying data pointers are fixed
  // addresses that CUDA graph replay will write to / read from.
  // SlimTensor pointers — owned by this handle.
  std::vector<void*> static_input_ptrs;  // raw GPU data pointers for inputs
  std::vector<void*> static_output_ptrs; // raw GPU data pointers for outputs
  std::vector<std::vector<int64_t>> static_input_sizes;
  std::vector<std::vector<int64_t>> static_input_strides;
  std::vector<std::vector<int64_t>> static_output_sizes;
  std::vector<std::vector<int64_t>> static_output_strides;
  std::vector<int> static_input_scalar_types;
  std::vector<int> static_output_scalar_types;
  std::vector<size_t> static_input_nbytes;
  std::vector<size_t> static_output_nbytes;

  ~CudaDelegateHandle() {
    if (cuda_graph_exec) {
      cudaGraphExecDestroy(cuda_graph_exec);
    }
    if (cuda_graph) {
      cudaGraphDestroy(cuda_graph);
    }
    // Only free input buffers — output buffers are owned by the AOTI runtime
    // (allocated during graph capture via the caching allocator).
    for (auto* ptr : static_input_ptrs) {
      if (ptr)
        cudaFree(ptr);
    }
  }
};

} // namespace cuda
} // namespace backends
} // namespace executorch
