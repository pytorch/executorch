/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Provider.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Per-PortableBackend Provider registry.
 *
 * v1 commits to explicit factory (no static init): the
 * PortableBackend constructor takes a ProviderSet (a list of
 * unique_ptr<Provider>); registration order determines RuntimeId.
 * Convention: index 0 is the CPU Provider (host-slot invariant, §4.1).
 *
 * See §4.5.1 of the design doc.
 */
class ProviderRegistry {
 public:
  // Construct with the full Provider set up front. Each Provider is
  // assigned a fresh RuntimeId equal to its index. Caches the
  // is_available_on_device() result for every Provider.
  explicit ProviderRegistry(
      std::vector<std::unique_ptr<Provider>> providers);

  // Non-copyable, movable.
  ProviderRegistry(const ProviderRegistry&) = delete;
  ProviderRegistry& operator=(const ProviderRegistry&) = delete;
  ProviderRegistry(ProviderRegistry&&) = default;
  ProviderRegistry& operator=(ProviderRegistry&&) = default;

  // The set of providers whose is_available_on_device() returned true.
  // Cached for the life of the registry. v1 does NOT support hot-plug.
  ::executorch::runtime::Span<Provider* const> available() const {
    return ::executorch::runtime::Span<Provider* const>(
        available_.data(), available_.size());
  }

  // All providers regardless of availability (for diagnostics).
  ::executorch::runtime::Span<Provider* const> all() const {
    return ::executorch::runtime::Span<Provider* const>(
        all_.data(), all_.size());
  }

  // Lookup by id.
  Provider* lookup(RuntimeId id) const {
    if (id == kHost) return nullptr;
    if (id >= owned_.size()) return nullptr;
    return owned_[id].get();
  }

 private:
  std::vector<std::unique_ptr<Provider>> owned_;  // index = RuntimeId
  std::vector<Provider*> all_;                    // raw pointers parallel to owned_
  std::vector<Provider*> available_;              // subset where is_available_on_device()
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
