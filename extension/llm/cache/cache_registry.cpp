/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cache_registry.h>

#include <atomic>

#include <executorch/runtime/core/error.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

CacheRegistry& CacheRegistry::global() {
  static CacheRegistry registry;
  return registry;
}

void CacheRegistry::install(
    const std::string& key,
    std::shared_ptr<Cache> cache) {
  std::lock_guard<std::mutex> lock(mu_);
  caches_[key] = std::move(cache);
}

std::shared_ptr<Cache> CacheRegistry::get(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = caches_.find(key);
  return it == caches_.end() ? nullptr : it->second;
}

void CacheRegistry::erase(const std::string& key) {
  std::lock_guard<std::mutex> lock(mu_);
  caches_.erase(key);
}

CacheFactory& CacheFactory::global() {
  static CacheFactory registry;
  return registry;
}

void CacheFactory::register_builder(
    const std::string& backend_id,
    const std::string& kind,
    CacheBuilder builder) {
  std::lock_guard<std::mutex> lock(mu_);
  builders_[{backend_id, kind}] = std::move(builder);
}

Result<std::shared_ptr<Cache>> CacheFactory::build(
    const std::string& backend_id,
    const std::string& kind,
    const CacheConfig& cfg) const {
  CacheBuilder builder;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = builders_.find({backend_id, kind});
    if (it == builders_.end()) {
      // Name what is registered. A kind is a string, so a typo is otherwise a
      // dead end. builders_ is ordered, so these come out sorted.
      std::string known;
      for (const auto& entry : builders_) {
        if (entry.first.first != backend_id) {
          continue;
        }
        if (!known.empty()) {
          known += ", ";
        }
        known += entry.first.second;
      }
      ET_LOG(
          Error,
          "no '%s' cache registered for '%s'; registered: %s",
          kind.c_str(),
          backend_id.c_str(),
          known.empty() ? "(none)" : known.c_str());
      return Error::NotFound;
    }
    builder = it->second;
  }
  // Checked here rather than in each cache: `layers` is indexed directly, so a
  // list that is neither size 1 nor n_layers reads past the end.
  ET_CHECK_OR_RETURN_ERROR(
      valid(cfg),
      InvalidArgument,
      "cache: invalid CacheConfig for %s:%s",
      backend_id.c_str(),
      kind.c_str());
  // A builder that hands back null would otherwise travel as an ok() Result
  // and be dereferenced by the caller.
  auto cache = builder(cfg);
  ET_CHECK_OR_RETURN_ERROR(
      cache != nullptr,
      Internal,
      "cache: builder for %s:%s returned null",
      backend_id.c_str(),
      kind.c_str());
  return cache;
}

std::string new_cache_key() {
  static std::atomic<uint64_t> counter{0};
  return "cache-" + std::to_string(counter.fetch_add(1));
}

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
