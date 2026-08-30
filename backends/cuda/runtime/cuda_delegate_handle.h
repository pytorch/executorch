/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace cuda {

using AOTInductorModelContainerGetConstantDtypeFunc =
    aoti::AOTIRuntimeError (*)(
        aoti::AOTInductorModelContainerHandle container_handle,
        size_t idx,
        int32_t* dtype);
struct CudaWeightStorage {
  void* data{nullptr};
  size_t nbytes{0};
  aoti::slim::c10::DeviceType device_type{aoti::slim::c10::DeviceType::CUDA};
  int device_index{0};

  CudaWeightStorage(
      void* data_,
      size_t nbytes_,
      aoti::slim::c10::DeviceType device_type_,
      int device_index_)
      : data(data_),
        nbytes(nbytes_),
        device_type(device_type_),
        device_index(device_index_) {}

  ~CudaWeightStorage() {
    if (data == nullptr) {
      return;
    }
    if (device_type == aoti::slim::c10::DeviceType::CPU) {
      std::free(data);
      return;
    }
    int previous_device = 0;
    const cudaError_t get_device_error = cudaGetDevice(&previous_device);
    if (get_device_error == cudaSuccess && previous_device != device_index) {
      (void)cudaSetDevice(device_index);
    }
    (void)cudaFree(data);
    if (get_device_error == cudaSuccess && previous_device != device_index) {
      (void)cudaSetDevice(previous_device);
    }
  }

  CudaWeightStorage(const CudaWeightStorage&) = delete;
  CudaWeightStorage& operator=(const CudaWeightStorage&) = delete;
};

// Phases of the CUDA graph lifecycle for a delegate handle.
//
// The transition flow is:
//   Disabled  ──(if CUDA graph is enabled for this method)──▶  Warmup
//   Warmup    ──(after `warmup_remaining` execute() calls)──▶  Replay
//
// - Disabled: CUDA graph is not used for this method. Every execute() runs
//   eagerly through the normal kernel-launch path.
//
// - Warmup:   The first `kCudaGraphWarmupSteps` execute() calls run eagerly
//   to let lazy allocators, autotuners, and JIT-compiled kernels stabilize.
//   On the final warmup step (`warmup_remaining == 0`), persistent static
//   input/output GPU buffers are allocated and the work is recorded into
//   `graph` / `graph_exec` via stream capture.
//
// - Replay:   The captured `graph_exec` is launched on every execute() call.
//   Inputs are memcpy'd into the static input buffers, the graph is replayed,
//   and outputs are memcpy'd back from the static output buffers. No tensor
//   setup or kernel launches happen on the host hot path.
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
      (void)cudaGraphExecDestroy(graph_exec);
    }
    if (graph) {
      (void)cudaGraphDestroy(graph);
    }
    // Only free input buffers — output buffers are owned by the AOTI runtime
    // (allocated during graph capture via the caching allocator).
    for (auto* ptr : static_input_ptrs) {
      if (ptr)
        (void)cudaFree(ptr);
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
        (void)cudaGraphExecDestroy(graph_exec);
      if (graph)
        (void)cudaGraphDestroy(graph);
      for (auto* ptr : static_input_ptrs) {
        if (ptr)
          (void)cudaFree(ptr);
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
  // Extra AOTI metadata used to validate per-FQN weights before binding.
  AOTInductorModelContainerGetConstantDtypeFunc get_constant_dtype{nullptr};

  // The per-thread stream. Nothing owns it: the value is a fixed sentinel the
  // driver resolves to a different stream on each host thread, so releasing the
  // holder destroys nothing.
  cudaStream_t cuda_stream = nullptr;

  // The stream this handle's work runs on.
  cudaStream_t get_cuda_stream() const {
    return cuda_stream;
  }

  // CUDA graph state (warmup, capture, replay, static buffers)
  CudaGraphState cuda_graph_state;

  // Per-FQN weight artifacts keep the allocations and their
  // SlimTensor handles alive for as long as AOTI may reference their views.
  std::vector<std::shared_ptr<CudaWeightStorage>> fqn_weight_storages;
  std::vector<std::unique_ptr<aoti::slim::SlimTensor>> fqn_weight_tensors;
};

} // namespace cuda
} // namespace backends
} // namespace executorch
