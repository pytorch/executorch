/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNWeightsCacheManager.h>

#include <executorch/runtime/core/error.h>

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
