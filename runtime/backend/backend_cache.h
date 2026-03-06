/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace runtime {

/**
 * Abstract key-value cache for backend init artifacts.
 *
 * Backends can persist expensive initialization results (packed weights,
 * compiled graphs, etc.) and restore them on subsequent runs, skipping the
 * processed data load entirely.
 *
 * Keys are arbitrary C strings chosen by the backend. The runtime wraps
 * each backend's view of the cache with DelegateBackendCache to prevent
 * key collisions between backends and delegates.
 *
 * All methods receive the full scoping triple (backend_id, delegate_index,
 * key) so implementations can organize storage however they choose, without
 * string concatenation in the runtime layer.
 */
class BackendCache {
 public:
  virtual ~BackendCache() = default;

  /// Load cached data. Returns Error::NotFound if key doesn't exist.
  virtual Result<FreeableBuffer> load(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      size_t alignment = alignof(std::max_align_t)) const = 0;

  /// Save data to cache. Overwrites existing entries.
  virtual Error save(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      const void* data,
      size_t size) = 0;

  /// Remove a cache entry. Returns Error::NotFound if key doesn't exist.
  virtual Error
  remove(const char* backend_id, size_t delegate_index, const char* key) = 0;
};

/**
 * Wraps a BackendCache, capturing backend_id and delegate_index so that
 * backends see a simple (key)-only API.
 *
 * Not a subclass of BackendCache -- this is a separate, simpler type that
 * forwards to the underlying cache with the scoping components attached.
 */
class DelegateBackendCache final {
 public:
  DelegateBackendCache(
      BackendCache* cache,
      const char* backend_id,
      size_t delegate_index)
      : cache_(cache),
        backend_id_(backend_id),
        delegate_index_(delegate_index) {}

  Result<FreeableBuffer> load(
      const char* key,
      size_t alignment = alignof(std::max_align_t)) const {
    return cache_->load(backend_id_, delegate_index_, key, alignment);
  }

  Error save(const char* key, const void* data, size_t size) {
    return cache_->save(backend_id_, delegate_index_, key, data, size);
  }

  Error remove(const char* key) {
    return cache_->remove(backend_id_, delegate_index_, key);
  }

 private:
  BackendCache* cache_;
  const char* backend_id_;
  size_t delegate_index_;
};

} // namespace runtime
} // namespace executorch
