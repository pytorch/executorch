/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Runtime.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

/**
 * Per-PortableBackend Runtime registry.
 *
 * v1 commits to explicit factory (no static init): the
 * PortableBackend constructor takes a ProviderSet (a list of
 * unique_ptr<Runtime>); registration order determines RuntimeId.
 * Convention: index 0 is the CPU Runtime (host-slot invariant, §4.1).
 *
 * See §4.5.1 of the design doc.
 */
class RuntimeRegistry {
 public:
  // Construct with the full Runtime set up front. Each Runtime is
  // assigned a fresh RuntimeId equal to its index. Caches the
  // is_available_on_device() result for every Runtime.
  explicit RuntimeRegistry(std::vector<std::unique_ptr<Runtime>> providers);

  // Non-copyable, movable.
  RuntimeRegistry(const RuntimeRegistry&) = delete;
  RuntimeRegistry& operator=(const RuntimeRegistry&) = delete;
  RuntimeRegistry(RuntimeRegistry&&) = default;
  RuntimeRegistry& operator=(RuntimeRegistry&&) = default;

  // The set of providers whose is_available_on_device() returned true.
  // Cached for the life of the registry. v1 does NOT support hot-plug.
  ::executorch::runtime::Span<Runtime* const> available() const {
    return ::executorch::runtime::Span<Runtime* const>(
        available_.data(), available_.size());
  }

  // All providers regardless of availability (for diagnostics).
  ::executorch::runtime::Span<Runtime* const> all() const {
    return ::executorch::runtime::Span<Runtime* const>(
        all_.data(), all_.size());
  }

  // Lookup by id.
  Runtime* lookup(RuntimeId id) const {
    if (id == kHost)
      return nullptr;
    if (id >= owned_.size())
      return nullptr;
    return owned_[id].get();
  }

 private:
  std::vector<std::unique_ptr<Runtime>> owned_; // index = RuntimeId
  std::vector<Runtime*> all_; // raw pointers parallel to owned_
  std::vector<Runtime*> available_; // subset where is_available_on_device()
};

} // namespace native
} // namespace backends
} // namespace executorch
