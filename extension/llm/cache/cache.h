/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Neutral, ET-independent ownership handle for an off-graph KV cache. The
// registry owns a cache as a CacheBase* -- an opaque, cache-agnostic anchor --
// and hands it back by key. The typed faces a runner and backend use to drive
// the cache are added by the concrete cache implementation.

namespace executorch {
namespace extension {
namespace llm {
namespace cache {

// Ownership / erasure anchor: the registry owns and deletes a cache through
// this base, staying agnostic to the concrete cache type.
class CacheBase {
 public:
  virtual ~CacheBase() = default;
};

// Model facts a cache factory builds from. capacity is the logical cap;
// n_layers is the number of attention layers; initial_capacity is the byte
// layer's starting pool size before it grows (by doubling) toward capacity.
struct CacheConfig {
  int capacity;
  int n_layers;
  int initial_capacity = 512;
};

} // namespace cache
} // namespace llm
} // namespace extension
} // namespace executorch
