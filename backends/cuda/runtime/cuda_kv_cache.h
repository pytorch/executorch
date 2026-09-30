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

#include <cuda_runtime.h>

#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {

namespace cache = ::executorch::extension::llm::cache;

struct OffGraphKVMetrics {
  int64_t logical_length{0};
  // Rows currently allocated for each flat layer.
  int64_t flat_capacity{0};
  // Times flat storage has been reallocated. Every growth moves the storage,
  // so this also identifies the addresses a compiled program was bound to.
  int64_t growth_count{0};
  int64_t allocated_bytes{0};
};

// Backend face of the off-graph KV cache: what the CUDA delegate needs from
// whatever cache the runner installed under its registry key.
//
// Named here rather than in cache.h for the same reason as MLXCache: a backend
// face speaks the backend's own types (here, AOTI delegate handles and CUDA
// streams) and the neutral header cannot know about them.
//
// The cache owns memory only. Lowering replaced kvcache::update_and_attend
// with index_copy_ + triton::sdpa over sequence-major (BSHD) storage declared
// at the maximum capacity. Every access is bounded by kv_len along the
// sequence, so a flat layer may be backed by fewer rows than declared and grow
// on demand; the cache points the program's storage constants at its current
// allocations.
class CudaKVCache {
 public:
  static constexpr const char* kFaceName = "cuda.CudaKVCache";

  virtual ~CudaKVCache() = default;

  // Load time. Discovers this program's KV storage constants. One cache serves
  // every method of a model, so this is called once per delegate handle.
  //
  // Answers whether this program carries off-graph storage at all: a method
  // that does not (an embedding or vision pass) has no use for the cache, and
  // the caller should stop associating it with one.
  virtual runtime::Result<bool> note_handle(CudaDelegateHandle* handle) = 0;
  virtual void forget_handle(CudaDelegateHandle* handle) = 0;

  // Execute time, after prepare_step. Points the program's storage constants
  // at the current allocations, which move whenever the cache grows.
  virtual runtime::Error rebind_for_execute(CudaDelegateHandle* handle) = 0;

  // Between steps. prepare_step() admits a step of write_length tokens and
  // grows the flat layers to fit it; commit_step() advances the logical length
  // past it. `stream` is the stream the step executes on: the growth copy and
  // the release of the old storage are ordered on it, behind every earlier
  // step that read that storage and ahead of the kernels that read the new.
  //
  // Named apart from the base's commit(SeqStepPlan), which applies one layer's
  // already-computed plan: these take a token count and speak for the whole
  // cache.
  virtual runtime::Error prepare_step(
      int64_t write_length,
      cudaStream_t stream) = 0;
  virtual runtime::Error commit_step(int64_t write_length) = 0;

  virtual OffGraphKVMetrics metrics() const = 0;
};

// Builder for cache::kind::kSingle. Returns null when the geometry or config is
// invalid, so CacheFactory reports the failure instead of the cache asserting
// during construction.
//
// Flat layers start at cfg.initial_capacity rows and grow geometrically up to
// cfg.capacity; ring layers hold window + max_write - 1 rows from the start.
// cfg.kv_dtype is an ExecuTorch ScalarType; the slim enum the storage uses
// shares its numbering, and unsupported values are rejected here.
std::shared_ptr<cache::Cache> make_cuda_sequence_kv_cache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg);

} // namespace executorch::backends::cuda
