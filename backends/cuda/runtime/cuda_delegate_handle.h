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
#include <executorch/extension/cuda/device_guard.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {

// Forward declarations for the off-graph KV cache the delegate may be given.
// The concrete types live in cuda_kv_cache.h, which includes this header.
namespace extension::llm::cache {
class Cache;
} // namespace extension::llm::cache

namespace backends {
namespace cuda {

class CudaKVCache;

// Where a method's inputs carry the number of tokens a step writes: an index
// into execute()'s inputs and a dimension of that tensor. Declared by the
// export side as the "offgraph_kv_step_width" compile spec, "input_index:dim".
struct OffGraphKVStepWidth {
  int input{0};
  int dim{0};
};

using AOTInductorModelContainerGetConstantDtypeFunc =
    aoti::AOTIRuntimeError (*)(
        aoti::AOTInductorModelContainerHandle container_handle,
        size_t idx,
        int32_t* dtype);
using AOTInductorModelContainerGetConstantDataSizeFunc =
    aoti::AOTIRuntimeError (*)(
        aoti::AOTInductorModelContainerHandle container_handle,
        size_t idx,
        size_t* data_size);
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
    // A destructor cannot report, and the guard logs a failed restore itself.
    const auto guard =
        ::executorch::extension::cuda::CUDAGuard::create(device_index);
    (void)guard;
    (void)cudaFree(data);
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

  // Allocations the graph makes and never frees -- the outputs AOTI allocates
  // while being captured. AutoFreeOnLaunch frees one launch's before the next
  // launch of the same graph, so the last launch's outlive a graph that is
  // dropped unless released with it. Outputs that alias other memory are not
  // graph allocations and are not listed.
  std::vector<void*> graph_allocations;

  CudaGraphState() = default;

  ~CudaGraphState() {
    release();
  }

  // Records the graph's outstanding allocations: every memory-allocation node
  // without a matching free node. Call once the graph is captured.
  void note_graph_allocations() {
    graph_allocations.clear();
#if !defined(EXECUTORCH_USE_HIP)
    size_t count = 0;
    if (graph == nullptr ||
        cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) {
      return;
    }
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
      return;
    }
    std::vector<void*> allocated;
    std::vector<void*> freed;
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type;
      if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
        continue;
      }
      if (type == cudaGraphNodeTypeMemAlloc) {
        cudaMemAllocNodeParams params{};
        if (cudaGraphMemAllocNodeGetParams(node, &params) == cudaSuccess) {
          allocated.push_back(params.dptr);
        }
      } else if (type == cudaGraphNodeTypeMemFree) {
        void* ptr = nullptr;
        if (cudaGraphMemFreeNodeGetParams(node, &ptr) == cudaSuccess) {
          freed.push_back(ptr);
        }
      }
    }
    for (void* ptr : allocated) {
      if (std::find(freed.begin(), freed.end(), ptr) == freed.end()) {
        graph_allocations.push_back(ptr);
      }
    }
#endif
  }

  // Frees the captured graph, the static inputs pinned for it, and the
  // allocations its last launch left outstanding (see graph_allocations).
  void release() {
    if (graph_exec) {
      (void)cudaGraphExecDestroy(graph_exec);
      graph_exec = nullptr;
    }
    if (graph) {
      (void)cudaGraphDestroy(graph);
      graph = nullptr;
    }
    for (auto* ptr : static_input_ptrs) {
      if (ptr)
        (void)cudaFree(ptr);
    }
    // cudaFree synchronises, so the last replay's reads of these are done.
    for (auto* ptr : graph_allocations) {
      (void)cudaFree(ptr);
    }
    graph_allocations.clear();
    static_input_ptrs.clear();
    static_output_ptrs.clear();
    static_input_nbytes.clear();
    static_output_nbytes.clear();
  }

  // Drops the captured graph so a new one is captured, for when memory the
  // graph baked in has moved. One eager step first: rebinding constants resets
  // AOTI's constant-fold state, and the fold the next run performs cannot run
  // inside a stream capture. The kernels are already loaded, so one is enough.
  void recapture() {
    release();
    phase = CudaGraphPhase::Warmup;
    warmup_remaining = 1;
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
        static_output_nbytes(std::move(other.static_output_nbytes)),
        graph_allocations(std::move(other.graph_allocations)) {
    other.graph = nullptr;
    other.graph_exec = nullptr;
  }

  CudaGraphState& operator=(CudaGraphState&& other) noexcept {
    if (this != &other) {
      release();

      phase = other.phase;
      warmup_remaining = other.warmup_remaining;
      graph = other.graph;
      graph_exec = other.graph_exec;
      static_input_ptrs = std::move(other.static_input_ptrs);
      static_output_ptrs = std::move(other.static_output_ptrs);
      static_input_nbytes = std::move(other.static_input_nbytes);
      static_output_nbytes = std::move(other.static_output_nbytes);
      graph_allocations = std::move(other.graph_allocations);

      other.graph = nullptr;
      other.graph_exec = nullptr;
    }
    return *this;
  }
};

// CUDA-specific delegate handle that extends AOTIDelegateHandle.
struct CudaDelegateHandle : public aoti::AOTIDelegateHandle {
  // AOTI's run() records a completion event on the stream and queries it at
  // the start of the next run. Captured into a CUDA graph, that record belongs
  // to the graph, so the first run() after the graph is dropped (to recapture
  // against moved KV storage) fails the query. The single-threaded entry point
  // skips the event, and a CUDA-graph method runs single-threaded by design.
  // Null when the .so does not export it; run() is used then.
  aoti::AOTInductorModelContainerRunFunc run_single_threaded{nullptr};

  // Extra AOTI metadata used to validate per-FQN weights before binding.
  AOTInductorModelContainerGetConstantDtypeFunc get_constant_dtype{nullptr};
  // Bytes a constant's compiled shape spans; the off-graph KV cache checks its
  // storage against it before binding.
  AOTInductorModelContainerGetConstantDataSizeFunc get_constant_data_size{
      nullptr};

  // The per-thread stream. Nothing owns it: the value is a fixed sentinel the
  // driver resolves to a different stream on each host thread, so releasing the
  // holder destroys nothing. Initialised to that sentinel rather than null,
  // because null is the legacy default stream, which is a different stream and
  // would silently drop the per-thread ordering this handle relies on.
  cudaStream_t cuda_stream = cudaStreamPerThread;

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

  // The off-graph KV cache the runner installed for this model, resolved from
  // the registry at init. Null for an in-graph model, which is the signal that
  // this program owns its KV state as ordinary (mutable) buffers.
  //
  // The shared_ptr is the handle's own claim on the cache, so the cache
  // outlives the delegate even if the runner drops its guard first; the raw
  // pointer is the backend face of that same object. Forward-declared rather
  // than included: cuda_kv_cache.h includes this header.
  std::shared_ptr<::executorch::extension::llm::cache::Cache> kv_cache_shared;
  CudaKVCache* kv_cache{nullptr};

  // Where this method's inputs carry the step width. The compiled program
  // cannot report it, because lowering replaced the cache op with kernels over
  // pre-bound memory.
  OffGraphKVStepWidth kv_step_width;

  // Compiled shape of each off-graph KV constant by FQN, from the serialized
  // FQN-weight metadata. AOTI reports only a constant's bytes, possibly
  // rounded up to 64, which cannot tell two nearby geometries apart.
  std::unordered_map<std::string, std::vector<int64_t>> offgraph_kv_sizes;
};

} // namespace cuda
} // namespace backends
} // namespace executorch
