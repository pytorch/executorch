/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {

namespace cache = ::executorch::extension::llm::cache;

// Geometry and sizing for one off-graph cache, in the neutral vocabulary.
// cache::CacheGeometry is positional: layer id is the index into `layers`.
struct OffGraphKVSettings {
  cache::CacheGeometry geometry;
  cache::CacheConfig config;
  aoti::slim::c10::ScalarType storage_dtype{
      aoti::slim::c10::ScalarType::BFloat16};

  size_t element_size() const {
    return aoti::slim::c10::elementSize(storage_dtype);
  }

  // Slots a ring layer needs to serve one step of up to max_write tokens: the
  // step writes all of them before attending, and its earliest query still
  // reads back window - 1 positions. Same formula as cache::RingPolicy and as
  // ring_physical_capacity() in triton/kernels/offgraph_kv.py.
  int64_t ring_capacity(const cache::LayerGeometry& layer) const {
    const int max_write =
        config.max_write ? *config.max_write : layer.policy.window;
    return static_cast<int64_t>(layer.policy.window) + max_write - 1;
  }
};

struct OffGraphKVMetrics {
  int64_t logical_length{0};
  int64_t flat_capacity{0};
  int64_t growth_count{0};
  int64_t allocated_bytes{0};
};

using OffGraphKVContext = uint64_t;
constexpr OffGraphKVContext kInvalidOffGraphKVContext = 0;

namespace detail {

OffGraphKVContext offgraph_kv_create_context(OffGraphKVSettings settings);
void offgraph_kv_destroy_context(OffGraphKVContext context);
void offgraph_kv_begin_load(OffGraphKVContext context);
void offgraph_kv_end_load();
runtime::Error offgraph_kv_validate(OffGraphKVContext context);
runtime::Error offgraph_kv_prepare(
    OffGraphKVContext context,
    int64_t write_length);
runtime::Error offgraph_kv_commit(
    OffGraphKVContext context,
    int64_t write_length);
runtime::Error offgraph_kv_reset(OffGraphKVContext context);
OffGraphKVMetrics offgraph_kv_metrics(OffGraphKVContext context);

} // namespace detail

class OffGraphKVCacheContextOwner final {
 public:
  class LoadScope final {
   public:
    explicit LoadScope(OffGraphKVContext context) {
      detail::offgraph_kv_begin_load(context);
    }
    ~LoadScope() {
      detail::offgraph_kv_end_load();
    }
    LoadScope(const LoadScope&) = delete;
    LoadScope& operator=(const LoadScope&) = delete;
    LoadScope(LoadScope&&) = delete;
    LoadScope& operator=(LoadScope&&) = delete;
  };

  explicit OffGraphKVCacheContextOwner(OffGraphKVSettings settings)
      : context_(detail::offgraph_kv_create_context(std::move(settings))) {}
  ~OffGraphKVCacheContextOwner() {
    detail::offgraph_kv_destroy_context(context_);
  }

  OffGraphKVCacheContextOwner(const OffGraphKVCacheContextOwner&) = delete;
  OffGraphKVCacheContextOwner& operator=(const OffGraphKVCacheContextOwner&) =
      delete;
  OffGraphKVCacheContextOwner(OffGraphKVCacheContextOwner&&) = delete;
  OffGraphKVCacheContextOwner& operator=(OffGraphKVCacheContextOwner&&) = delete;

  template <typename F>
  decltype(auto) with_load_scope(F&& fn) const {
    LoadScope scope(context_);
    return std::forward<F>(fn)();
  }

  runtime::Error validate() const {
    return detail::offgraph_kv_validate(context_);
  }
  runtime::Error prepare(int64_t write_length) const {
    return detail::offgraph_kv_prepare(context_, write_length);
  }
  runtime::Error commit(int64_t write_length) const {
    return detail::offgraph_kv_commit(context_, write_length);
  }
  runtime::Error reset() const {
    return detail::offgraph_kv_reset(context_);
  }
  OffGraphKVMetrics metrics() const {
    return detail::offgraph_kv_metrics(context_);
  }

 private:
  OffGraphKVContext context_{kInvalidOffGraphKVContext};
};

void offgraph_kv_note_handle(CudaDelegateHandle* handle);
void offgraph_kv_forget_handle(CudaDelegateHandle* handle);
runtime::Error offgraph_kv_rebind_for_execute(CudaDelegateHandle* handle);

} // namespace executorch::backends::cuda
