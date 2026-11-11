/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <executorch/extension/llm/batching/types.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

// Application-owned immutable hints; never part of prefix identity or cloning.
using CheckpointMetadata = std::shared_ptr<const void>;

struct ET_EXPERIMENTAL PrefixCacheEntryView {
  std::reference_wrapper<const PrefixIdentity> identity;
  CheckpointMetadata metadata;
};

using PrefixCacheEntries = std::vector<PrefixCacheEntryView>;

// On full-cache insertion, return a victim's original MRU index or nullopt.
// Invalid indices and exceptions decline insertion. Identities are borrowed
// for the call; do not retain them or reenter the cache.
using PrefixCacheRetentionPolicy = std::function<std::optional<std::size_t>(
    const PrefixCacheEntryView& incoming,
    const PrefixCacheEntries& entries)>;

// Global LRU, with no protected early checkpoints.
inline std::optional<std::size_t> lru_prefix_cache_retention(
    const PrefixCacheEntryView&,
    const PrefixCacheEntries& entries) {
  if (entries.empty()) {
    return std::nullopt;
  }
  return entries.size() - 1;
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
