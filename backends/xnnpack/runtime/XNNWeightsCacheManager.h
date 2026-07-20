/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace executorch::backends::xnnpack {

/**
 * One `XNNWeightsCache` per cache file path. Mirrors
 * `XNNWorkspaceManager`'s PerModel pattern with `weak_ptr` so
 * instances live as long as the executors owning them.
 *
 * Per-path keying prevents `initialize_for_runtime` from a second
 * path tearing down the first path's fd / mmap regions (SIGBUS).
 *
 * Empty path returns one shared heap-only instance so callers
 * without a file still get XNNPACK's in-memory name dedup.
 *
 * Lock order: `meta_mutex_` → `XNNWeightsCache::mutex()` →
 * `XNNWorkspaceManager::workspace_meta_mutex_` → `XNNWorkspace::mutex_`.
 */
class XNNWeightsCacheManager {
 public:
  XNNWeightsCacheManager() = default;
  ~XNNWeightsCacheManager() = default;

  XNNWeightsCacheManager(const XNNWeightsCacheManager&) = delete;
  XNNWeightsCacheManager& operator=(const XNNWeightsCacheManager&) = delete;
  XNNWeightsCacheManager(XNNWeightsCacheManager&&) = delete;
  XNNWeightsCacheManager& operator=(XNNWeightsCacheManager&&) = delete;

  /** Shared `XNNWeightsCache` for `cache_file_path`. Empty path
   * returns one shared heap-only instance. Never null on success. */
  runtime::Result<std::shared_ptr<delegate::XNNWeightsCache>> get_or_create(
      const std::string& cache_file_path);

  /** Walk live caches and call `save_packed_index()` on each under
   * its per-instance mutex. Returns the first error; keeps going so
   * one failure doesn't strand the others. Opportunistically erases
   * expired weak_ptrs. */
  runtime::Error save_all();

  /** Test-only: count of live (non-expired) entries. */
  size_t live_count() const;

 private:
  mutable std::mutex meta_mutex_;
  std::unordered_map<std::string, std::weak_ptr<delegate::XNNWeightsCache>>
      caches_;

  // Separate slot for the empty-path (heap-only) cache to avoid
  // string-hashing and contention with mmap-path callers.
  mutable std::mutex empty_path_mutex_;
  std::weak_ptr<delegate::XNNWeightsCache> empty_path_cache_;
};

} // namespace executorch::backends::xnnpack
