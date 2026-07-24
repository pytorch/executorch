/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Install / rendezvous machinery for the off-graph KV cache. The DelegateHandle
// is opaque to the host, so the runner (which knows the cache kind) creates the
// cache and binds it to the delegate through a process-global registry; the two
// sides rendezvous on a cache_key passed as a runtime backend-load option.
// Neutral: everything here is expressed over Cache*.

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <executorch/extension/llm/cache/cache.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

// Process-global map<cache_key, shared_ptr<Cache>>. Ownership is shared: the
// registry entry, the runner's session guard, and the delegate handle all hold
// the cache, so erasing the entry mid-method is safe.
class CacheRegistry {
 public:
  static CacheRegistry& global();

  void install(const std::string& key, std::shared_ptr<Cache> cache);
  std::shared_ptr<Cache> get(const std::string& key) const;
  void erase(const std::string& key);

 private:
  CacheRegistry() = default;

  mutable std::mutex mu_;
  std::unordered_map<std::string, std::shared_ptr<Cache>> caches_;
};

// Cache kind is expressed by which factory you call: backends register a
// builder per (backend_id, kind) and the kind survives only as an internal
// lookup tag.
using CacheBuilder = std::function<std::shared_ptr<Cache>(const CacheConfig&)>;

class CacheBuilderRegistry {
 public:
  static CacheBuilderRegistry& global();

  void register_builder(
      const std::string& backend_id,
      const std::string& kind,
      CacheBuilder builder);
  // Throws if no builder is registered for (backend_id, kind).
  std::shared_ptr<Cache> build(
      const std::string& backend_id,
      const std::string& kind,
      const CacheConfig& cfg) const;

 private:
  CacheBuilderRegistry() = default;

  mutable std::mutex mu_;
  std::unordered_map<std::string, CacheBuilder>
      builders_; // backend_id + ":" + kind
};

// Process-global atomic counter -> "cache-N"; centralizes key generation so
// keys never collide.
std::string make_unique_key();

// RAII: installs the cache into the global registry under a unique key on
// construction and erases it on destruction (no leak on any exit path). Holds
// the runner's shared_ptr and exposes the typed face. Face = Cache or a derived
// interface (e.g. a future SeqCache / TreeCache).
template <class Face>
class CacheSession {
 public:
  CacheSession(std::string key, std::shared_ptr<Face> cache)
      : key_(std::move(key)), cache_(std::move(cache)) {
    CacheRegistry::global().install(key_, cache_);
  }
  ~CacheSession() {
    CacheRegistry::global().erase(key_);
  }

  CacheSession(const CacheSession&) = delete;
  CacheSession& operator=(const CacheSession&) = delete;

  Face* operator->() const {
    return cache_.get();
  }
  const std::string& key() const {
    return key_;
  }

 private:
  std::string key_;
  std::shared_ptr<Face> cache_;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
