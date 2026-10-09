/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/extension/cuda/runtime_api.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::cuda {

// The bytes behind an off-graph KV cache on one CUDA device: per-layer K/V row
// storage, plus fixed-size side buffers a layout may declare alongside it, and
// the AOTI bindings that point every compiled program at them. Which rows mean
// what is the owning cache's business, never this class's.
//
// Growable layers share one row count that grows geometrically toward their
// declared rows, moving their storage; fixed layers and side buffers are
// allocated once and never move, so a program may capture their addresses.
//
// Not thread-safe: the owning cache serializes every call.
class CudaKVPool final {
 public:
  struct Layer {
    int64_t n_kv_heads{0};
    int64_t head_dim{0};
    // Rows the compiled program declares for this layer's storage.
    int64_t declared_rows{0};
    bool growable{false};
  };

  // A constant the program declares by FQN with a fixed, contiguous shape.
  struct SideBuffer {
    std::string fqn;
    ::executorch::backends::aoti::slim::c10::ScalarType dtype{
        ::executorch::backends::aoti::slim::c10::ScalarType::Byte};
    std::vector<int64_t> sizes;
  };

  CudaKVPool(
      std::vector<Layer> layers,
      std::vector<SideBuffer> side_buffers,
      ::executorch::backends::aoti::slim::c10::ScalarType storage_dtype,
      int64_t initial_rows);

  CudaKVPool(const CudaKVPool&) = delete;
  CudaKVPool& operator=(const CudaKVPool&) = delete;
  CudaKVPool(CudaKVPool&&) = delete;
  CudaKVPool& operator=(CudaKVPool&&) = delete;

  ~CudaKVPool();

  // Load time. Discovers this program's storage constants; false = it carries
  // none. A program carrying some layers but not all, or layers without every
  // side buffer, is rejected as disagreeing with the pool's geometry.
  runtime::Result<bool> note_handle(CudaDelegateHandle* handle);
  void forget_handle(CudaDelegateHandle* handle);
  bool serves(CudaDelegateHandle* handle) const;

  // Whole-model check: every layer and side buffer was found in some program.
  runtime::Error validate() const;

  // Orders `stream` after the previous step, then makes the growable layers
  // hold at least `required_rows`, carrying their first `live_rows` rows
  // across a growth. The first call also allocates the fixed storage.
  runtime::Error
  prepare(int64_t required_rows, int64_t live_rows, cudaStream_t stream);

  // Records where the step's work ends on its stream, for the next step to
  // follow if it runs on another.
  runtime::Error mark_step_done();

  // Points `handle`'s constants at the current storage. Cached until storage
  // moves. Precondition: allocated().
  runtime::Error bind(CudaDelegateHandle* handle);

  bool allocated() const {
    return allocated_;
  }
  // Device pointer of side buffer `index`, fixed for the pool's life once
  // allocated().
  void* side_buffer(size_t index) const;
  size_t side_buffer_bytes(size_t index) const;
  // The stream of the latest step, on which side-buffer writes must be issued.
  cudaStream_t stream() const {
    return stream_;
  }

  int64_t rows() const {
    return rows_;
  }
  int64_t allocated_bytes() const {
    return allocated_bytes_;
  }
  int64_t growth_count() const {
    return growth_count_;
  }

 private:
  struct Allocation {
    void* k{nullptr};
    void* v{nullptr};
    int64_t rows{0};
  };

  enum class Slot { Key, Value, Side };

  struct Descriptor {
    std::string internal_name;
    Slot slot{Slot::Key};
    size_t index{0};
  };

  // The tensors a handle's AOTI constants currently point at.
  struct Bound {
    std::vector<std::unique_ptr<::executorch::backends::aoti::slim::SlimTensor>>
        tensors;
  };

  size_t row_bytes(const Layer& layer) const;
  static std::vector<int64_t> layer_sizes(const Layer& layer);
  runtime::Error allocate_layer(
      const Layer& layer,
      int64_t rows,
      cudaStream_t stream,
      Allocation& out);
  void release(void* ptr, cudaStream_t stream);
  void discard(const Layer& layer, Allocation& allocation, cudaStream_t stream);
  runtime::Error follow_previous_step(cudaStream_t stream);
  runtime::Error allocate_initial(int64_t required_rows, cudaStream_t stream);
  runtime::Error allocate_side_buffers(cudaStream_t stream);
  runtime::Error grow(int64_t new_rows, int64_t live_rows, cudaStream_t stream);
  runtime::Error build_descriptors(CudaDelegateHandle* handle);
  runtime::Error check_compiled(
      CudaDelegateHandle* handle,
      size_t constant_index,
      const std::string& name,
      ::executorch::backends::aoti::slim::c10::ScalarType dtype,
      const std::vector<int64_t>& sizes) const;

  std::vector<Layer> layers_;
  std::vector<SideBuffer> side_specs_;
  ::executorch::backends::aoti::slim::c10::ScalarType storage_dtype_;
  int64_t initial_rows_;
  int64_t max_rows_{0};

  int device_{0};
  bool device_known_{false};
  bool allocated_{false};
  // The stream of the latest step, and so of the latest use of the storage.
  cudaStream_t stream_{cudaStreamPerThread};
  // Recorded on stream_ when a step commits; a later step on another stream
  // waits on it before touching the storage.
  cudaEvent_t last_step_done_{nullptr};

  int64_t rows_{0};
  int64_t allocated_bytes_{0};
  int64_t growth_count_{0};
  std::vector<Allocation> allocations_;
  std::vector<void*> side_buffers_;

  std::unordered_map<CudaDelegateHandle*, std::vector<Descriptor>> descriptors_;
  std::unordered_map<CudaDelegateHandle*, Bound> bound_;
  std::unordered_set<std::string> discovered_fqns_;
};

// The FQN the lowering pass gives a layer's K or V storage constant.
std::string offgraph_kv_layer_fqn(int64_t layer_id, const char* suffix);

} // namespace executorch::backends::cuda
