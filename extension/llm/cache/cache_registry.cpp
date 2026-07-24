/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cache_registry.h>

#include <atomic>
#include <stdexcept>

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

CacheBuilderRegistry& CacheBuilderRegistry::global() {
  static CacheBuilderRegistry registry;
  return registry;
}

void CacheBuilderRegistry::register_builder(
    const std::string& backend_id,
    const std::string& kind,
    CacheBuilder builder) {
  std::lock_guard<std::mutex> lock(mu_);
  builders_[backend_id + ":" + kind] = std::move(builder);
}

std::shared_ptr<Cache> CacheBuilderRegistry::build(
    const std::string& backend_id,
    const std::string& kind,
    const CacheConfig& cfg) const {
  CacheBuilder builder;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = builders_.find(backend_id + ":" + kind);
    if (it == builders_.end()) {
      throw std::runtime_error(
          "no cache builder registered for " + backend_id + ":" + kind);
    }
    builder = it->second;
  }
  return builder(cfg);
}

std::string make_unique_key() {
  static std::atomic<uint64_t> counter{0};
  return "cache-" + std::to_string(counter.fetch_add(1));
}

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
