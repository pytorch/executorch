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
  const PrefixIdentity& identity;
  const CheckpointMetadata& metadata;
};

using PrefixCacheEntries = std::vector<PrefixCacheEntryView>;

// Called before a distinct insertion into a full cache. Entries are ordered
// most recently used first and borrowed for the call. Return an existing
// entry's index to evict, or nullopt to decline the incoming snapshot.
// An invalid index or exception also declines it, leaving entries intact.
// Must not reenter the cache or retain the borrowed references.
using PrefixCacheRetentionPolicy = std::function<std::optional<std::size_t>(
    const PrefixCacheEntryView& incoming,
    const PrefixCacheEntries& entries)>;

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
