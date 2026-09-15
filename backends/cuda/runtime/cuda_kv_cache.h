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
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::cuda {

enum class OffGraphKVPolicy { Flat, Ring };

struct OffGraphKVLayerConfig {
  int64_t layer_id{0};
  OffGraphKVPolicy policy{OffGraphKVPolicy::Flat};
  int64_t window{0};
  int64_t num_kv_heads{0};
  int64_t head_dim{0};
};

struct OffGraphKVConfig {
  int64_t maximum_capacity{0};
  int64_t initial_capacity{0};
  aoti::slim::c10::ScalarType storage_dtype{
      aoti::slim::c10::ScalarType::BFloat16};
  std::vector<OffGraphKVLayerConfig> layers;

  size_t element_size() const {
    return aoti::slim::c10::elementSize(storage_dtype);
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
constexpr int kNoOffGraphKVSession = -1;

namespace detail {

OffGraphKVContext offgraph_kv_create_context(OffGraphKVConfig config);
void offgraph_kv_destroy_context(OffGraphKVContext context);
void offgraph_kv_begin_load(OffGraphKVContext context);
void offgraph_kv_end_load();
runtime::Error offgraph_kv_validate(OffGraphKVContext context);
runtime::Result<int> offgraph_kv_create_session(OffGraphKVContext context);
void offgraph_kv_destroy_session(OffGraphKVContext context, int token);
void offgraph_kv_set_active(OffGraphKVContext context, int token);
runtime::Error offgraph_kv_prepare(
    OffGraphKVContext context,
    int token,
    int64_t write_length);
runtime::Error offgraph_kv_commit(
    OffGraphKVContext context,
    int token,
    int64_t write_length);
runtime::Error offgraph_kv_reset(OffGraphKVContext context, int token);
OffGraphKVMetrics offgraph_kv_metrics(OffGraphKVContext context);
int64_t offgraph_kv_initial_bytes_per_session(OffGraphKVContext context);

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

  class ActiveSessionScope final {
   public:
    ActiveSessionScope(OffGraphKVContext context, int token) {
      detail::offgraph_kv_set_active(context, token);
    }
    ~ActiveSessionScope() {
      detail::offgraph_kv_set_active(
          kInvalidOffGraphKVContext, kNoOffGraphKVSession);
    }
    ActiveSessionScope(const ActiveSessionScope&) = delete;
    ActiveSessionScope& operator=(const ActiveSessionScope&) = delete;
    ActiveSessionScope(ActiveSessionScope&&) = delete;
    ActiveSessionScope& operator=(ActiveSessionScope&&) = delete;
  };

  explicit OffGraphKVCacheContextOwner(OffGraphKVConfig config)
      : context_(detail::offgraph_kv_create_context(std::move(config))) {}
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
  runtime::Result<int> create_session() const {
    return detail::offgraph_kv_create_session(context_);
  }
  void destroy_session(int token) const {
    detail::offgraph_kv_destroy_session(context_, token);
  }
  template <typename F>
  decltype(auto) with_active_session(int token, F&& fn) const {
    ActiveSessionScope scope(context_, token);
    return std::forward<F>(fn)();
  }
  runtime::Error prepare(int token, int64_t write_length) const {
    return detail::offgraph_kv_prepare(context_, token, write_length);
  }
  runtime::Error commit(int token, int64_t write_length) const {
    return detail::offgraph_kv_commit(context_, token, write_length);
  }
  runtime::Error reset(int token) const {
    return detail::offgraph_kv_reset(context_, token);
  }
  OffGraphKVMetrics metrics() const {
    return detail::offgraph_kv_metrics(context_);
  }
  int64_t initial_bytes_per_session() const {
    return detail::offgraph_kv_initial_bytes_per_session(context_);
  }

 private:
  OffGraphKVContext context_{kInvalidOffGraphKVContext};
};

void offgraph_kv_note_handle(CudaDelegateHandle* handle);
void offgraph_kv_forget_handle(CudaDelegateHandle* handle);
runtime::Error offgraph_kv_rebind_for_execute(CudaDelegateHandle* handle);

} // namespace executorch::backends::cuda
