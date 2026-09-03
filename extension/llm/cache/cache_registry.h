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
// Caches are owned as Cache*; a face comes from as<T>(), null when the cache
// does not implement it.

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

// Backend-load option carrying the key of an installed cache. This name is the
// rendezvous contract shared by cache-owning runners and cache-aware backends.
inline constexpr char kCacheKeyOption[] = "llm_cache_registry_key";

// Process-global map<cache_key, shared_ptr<Cache>>. Ownership is shared:
// the registry entry, the runner's guard, and the delegate handle all hold
// the cache, so erasing the entry mid-method is safe.
class ET_EXPERIMENTAL CacheRegistry {
 public:
  static CacheRegistry& global();

  // The delegate's half of the rendezvous: resolve a key it was handed as a
  // backend option. Null if no cache is published under it.
  std::shared_ptr<Cache> get(const std::string& key) const;

 private:
  CacheRegistry() = default;

  // Only InstallGuard may publish, so an entry cannot outlive its owner, two
  // callers cannot collide on a key, and no erase can go unpaired.
  friend class InstallGuard;
  void install(const std::string& key, std::shared_ptr<Cache> cache);
  void erase(const std::string& key);

  mutable std::mutex mu_;
  std::unordered_map<std::string, std::shared_ptr<Cache>> caches_;
};

// The registered cache kinds. Spelling one inline is a runtime NotFound rather
// than a compile error, so go through these.
namespace kind {
// One sequence over per-layer runs.
inline constexpr const char* kSingle = "single";
// Many sequences sharing one pool of per-token cells.
inline constexpr const char* kBatchedCell = "batched-cell";
} // namespace kind

// Cache kind is expressed by which factory you call: backends register a
// builder per (backend_id, kind) and the kind survives only as an internal
// lookup tag.
using CacheBuilder = std::function<std::shared_ptr<Cache>(const CacheConfig&)>;

class ET_EXPERIMENTAL CacheFactory {
 public:
  static CacheFactory& global();

  // Public so a test can hold its own rather than registering builders into
  // the process-global one, where they outlive it.
  CacheFactory() = default;

  // Registers one builder without replacing an existing entry. Returns
  // InvalidArgument if builder is empty or the pair is already registered.
  ET_NODISCARD Error register_builder(
      const std::string& backend_id,
      const std::string& kind,
      CacheBuilder builder);
  // Returns NotFound if no builder is registered for (backend_id, kind), and
  // Internal if the registered builder returns null.
  Result<std::shared_ptr<Cache>> build(
      const std::string& backend_id,
      const std::string& kind,
      const CacheConfig& cfg) const;

 private:
  mutable std::mutex mu_;
  std::map<std::pair<std::string, std::string>, CacheBuilder> builders_;
};

// RAII over one registry entry: installs the cache under a key of its own
// making on construction and erases it on destruction (no leak on any exit
// path). Minting the key here rather than taking one means two live guards
// cannot collide on it. Must outlive the load_method() whose backend init
// resolves the key.
//
// Destruction removes discoverability only. A shared_ptr already returned by
// CacheRegistry::get() remains valid independently.
class ET_EXPERIMENTAL InstallGuard {
 public:
  explicit InstallGuard(std::shared_ptr<Cache> cache);
  ~InstallGuard();

  InstallGuard(const InstallGuard&) = delete;
  InstallGuard& operator=(const InstallGuard&) = delete;

  // Adds the complete cache rendezvous option. BackendOptions copies the key
  // and value, so the resulting option remains valid independently.
  template <size_t N>
  Error set_option(::executorch::runtime::BackendOptions<N>& options) const {
    return options.set_option(kCacheKeyOption, key_.c_str());
  }

 private:
  std::string key_;
  std::shared_ptr<Cache> cache_;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
