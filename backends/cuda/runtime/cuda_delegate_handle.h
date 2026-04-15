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

enum class CudaGraphPhase {
  Disabled = 0,
  Warmup = 1,
  Replay = 2,
};

// All CUDA graph related state grouped into a single struct.
struct CudaGraphState {
  CudaGraphPhase phase = CudaGraphPhase::Disabled;
  int warmup_remaining = 0;

  // Captured graph and executable instance
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  // Static input/output GPU buffers pinned during capture.
  // These hold the tensor metadata; the underlying data pointers are fixed
  // addresses that CUDA graph replay will write to / read from.
  std::vector<void*> static_input_ptrs;
  std::vector<void*> static_output_ptrs;
  std::vector<size_t> static_input_nbytes;
  std::vector<size_t> static_output_nbytes;

  CudaGraphState() = default;

  ~CudaGraphState() {
    if (graph_exec) {
      cudaGraphExecDestroy(graph_exec);
    }
    if (graph) {
      cudaGraphDestroy(graph);
    }
    // Only free input buffers — output buffers are owned by the AOTI runtime
    // (allocated during graph capture via the caching allocator).
    for (auto* ptr : static_input_ptrs) {
      if (ptr)
        cudaFree(ptr);
    }
  }

  // Non-copyable: prevent double-free of CUDA resources
  CudaGraphState(const CudaGraphState&) = delete;
  CudaGraphState& operator=(const CudaGraphState&) = delete;

  // Movable
  CudaGraphState(CudaGraphState&& other) noexcept
      : phase(other.phase),
        warmup_remaining(other.warmup_remaining),
        graph(other.graph),
        graph_exec(other.graph_exec),
        static_input_ptrs(std::move(other.static_input_ptrs)),
        static_output_ptrs(std::move(other.static_output_ptrs)),
        static_input_nbytes(std::move(other.static_input_nbytes)),
        static_output_nbytes(std::move(other.static_output_nbytes)) {
    other.graph = nullptr;
    other.graph_exec = nullptr;
  }

  CudaGraphState& operator=(CudaGraphState&& other) noexcept {
    if (this != &other) {
      // Clean up existing resources
      if (graph_exec)
        cudaGraphExecDestroy(graph_exec);
      if (graph)
        cudaGraphDestroy(graph);
      for (auto* ptr : static_input_ptrs) {
        if (ptr)
          cudaFree(ptr);
      }

      phase = other.phase;
      warmup_remaining = other.warmup_remaining;
      graph = other.graph;
      graph_exec = other.graph_exec;
      static_input_ptrs = std::move(other.static_input_ptrs);
      static_output_ptrs = std::move(other.static_output_ptrs);
      static_input_nbytes = std::move(other.static_input_nbytes);
      static_output_nbytes = std::move(other.static_output_nbytes);

      other.graph = nullptr;
      other.graph_exec = nullptr;
    }
    return *this;
  }
};

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

  // CUDA graph state (warmup, capture, replay, static buffers)
  CudaGraphState cuda_graph_state;
};

} // namespace cuda
} // namespace backends
} // namespace executorch
