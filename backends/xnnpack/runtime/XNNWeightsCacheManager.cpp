/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNWeightsCacheManager.h>

#include <executorch/runtime/core/error.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace executorch::backends::xnnpack {

using executorch::runtime::Error;
using executorch::runtime::Result;

Result<std::shared_ptr<delegate::XNNWeightsCache>>
XNNWeightsCacheManager::get_or_create(const std::string& cache_file_path) {
  // Empty path → one shared heap-only instance. See header for why.
  if (cache_file_path.empty()) {
    std::scoped_lock<std::mutex> lock(empty_path_mutex_);
    if (auto live = empty_path_cache_.lock()) {
      return live;
    }
    auto cache = std::make_shared<delegate::XNNWeightsCache>();
    empty_path_cache_ = cache;
    return cache;
  }

  std::scoped_lock<std::mutex> lock(meta_mutex_);
  auto it = caches_.find(cache_file_path);
  if (it != caches_.end()) {
    if (auto live = it->second.lock()) {
      return live;
    }
    caches_.erase(it);
  }

  auto cache = std::make_shared<delegate::XNNWeightsCache>();
  // Set path before publishing into the map so concurrent callers
  // observe a fully initialized instance.
  cache->set_packed_cache_path(cache_file_path);
  caches_[cache_file_path] = cache;
  return cache;
}

Error XNNWeightsCacheManager::save_all() {
  // Snapshot live shared_ptrs under meta_mutex_, then release it
  // before per-instance save (honors lock order, lets get_or_create
  // on unrelated paths proceed during the save walk).
  std::vector<std::shared_ptr<delegate::XNNWeightsCache>> live;
  {
    std::scoped_lock<std::mutex> lock(meta_mutex_);
    live.reserve(caches_.size());
    for (auto it = caches_.begin(); it != caches_.end();) {
      if (auto cache = it->second.lock()) {
        live.push_back(std::move(cache));
        ++it;
      } else {
        it = caches_.erase(it);
      }
    }
  }

  Error first_err = Error::Ok;
  for (auto& cache : live) {
    std::lock_guard<std::mutex> lock(cache->mutex());
    Error err = cache->save_packed_index();
    if (err != Error::Ok && first_err == Error::Ok) {
      first_err = err;
    }
  }
  return first_err;
}

xnnpack::PackedCacheReport XNNWeightsCacheManager::report() const {
  // Snapshot path + instance under the owning mutexes, then read the counters
  // without XNNWeightsCache::mutex(). That mutex is held across the whole of
  // xnn_create_runtime, so waiting on it here would let a telemetry read stall
  // an inference thread for the length of a model compile.
  std::vector<
      std::pair<std::string, std::shared_ptr<delegate::XNNWeightsCache>>>
      live;
  {
    std::scoped_lock<std::mutex> lock(meta_mutex_);
    live.reserve(caches_.size());
    for (const auto& entry : caches_) {
      if (auto cache = entry.second.lock()) {
        live.emplace_back(entry.first, std::move(cache));
      }
    }
  }
  {
    std::scoped_lock<std::mutex> lock(empty_path_mutex_);
    if (auto cache = empty_path_cache_.lock()) {
      live.emplace_back(std::string{}, std::move(cache));
    }
  }
  // caches_ is unordered; sort so the report and the index into it are stable
  // across calls.
  std::sort(live.begin(), live.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  xnnpack::PackedCacheReport out;
  out.per_cache.reserve(live.size());
  for (const auto& [path, cache] : live) {
    out.per_cache.push_back(xnnpack::PackedCacheEntry{path, cache->stats()});
  }

  int64_t best_heap = -1;
  int32_t first_fallback = -1;
  for (size_t i = 0; i < out.per_cache.size(); ++i) {
    const auto& s = out.per_cache[i].stats;
    auto& agg = out.aggregate;
    agg.file_bytes += s.file_bytes;
    agg.heap_bytes += s.heap_bytes;
    agg.mapped_bytes += s.mapped_bytes;
    for (size_t r = 0; r < s.heap_bytes_by_reason.size(); ++r) {
      agg.heap_bytes_by_reason[r] += s.heap_bytes_by_reason[r];
    }
    // A fallback anywhere is the reportable outcome: if any cache on this
    // process went to heap, the process is carrying that memory.
    if (s.state == delegate::PackedCacheState::HeapFallback) {
      agg.state = s.state;
      if (first_fallback < 0) {
        first_fallback = static_cast<int32_t>(i);
      }
    } else if (
        s.state == delegate::PackedCacheState::FileBacked &&
        agg.state == delegate::PackedCacheState::Disabled) {
      agg.state = s.state;
    }
    if (s.heap_bytes > best_heap) {
      best_heap = s.heap_bytes;
      if (s.heap_bytes > 0) {
        out.dominant_fallback = static_cast<int32_t>(i);
      }
    }
  }
  // A cache can fail before it ever allocates, so fall back to the first
  // cache in HeapFallback rather than reporting no explanation at all.
  if (out.dominant_fallback < 0) {
    out.dominant_fallback = first_fallback;
  }

  // Argmax over per-reason totals summed across caches — not the local reason
  // of whichever cache allocated the most, which can disagree with the global
  // picture when one cache mixes reasons.
  int64_t worst = 0;
  for (size_t r = 0; r < out.aggregate.heap_bytes_by_reason.size(); ++r) {
    if (r == static_cast<size_t>(delegate::PackedCacheHeapReason::NotOptedIn)) {
      continue; // intended heap use, not a fallback
    }
    if (out.aggregate.heap_bytes_by_reason[r] > worst) {
      worst = out.aggregate.heap_bytes_by_reason[r];
      out.aggregate.heap_reason =
          static_cast<delegate::PackedCacheHeapReason>(r);
    }
  }
  return out;
}

delegate::PackedCacheStats XNNWeightsCacheManager::aggregate_stats() const {
  return report().aggregate;
}

size_t XNNWeightsCacheManager::live_count() const {
  std::scoped_lock<std::mutex> lock(meta_mutex_);
  size_t count = 0;
  for (const auto& entry : caches_) {
    if (!entry.second.expired()) {
      ++count;
    }
  }
  return count;
}

} // namespace executorch::backends::xnnpack
