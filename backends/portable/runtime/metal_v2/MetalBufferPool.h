/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalBufferPool - LRU buffer pool with best-fit matching
// Reuses MTLBuffer instances across stream->alloc()/free() cycles to avoid
// repeatedly hitting [device newBufferWithLength:] (~10µs). Best-fit lookup
// with bounded headroom (won't reuse a 2× larger buffer to avoid waste).
// Bounded total cache (default 256 MB) with LRU eviction.
// Extracted from MetalStream.h — definition unchanged.
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <cstddef>
#include <list>
#include <map>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalBufferPool {
public:
  explicit MetalBufferPool(id<MTLDevice> device, size_t maxBytes = 256 * 1024 * 1024);
  ~MetalBufferPool();

  /// Acquire a buffer of at least `size` bytes
  id<MTLBuffer> acquire(size_t size);

  /// Return buffer to pool
  void release(id<MTLBuffer> buffer);

  /// Clear all cached buffers
  void clear();

  /// Current bytes in pool
  size_t cachedBytes() const { return cachedBytes_; }

  /// Maximum bytes the pool will hold before evicting LRU entries.
  size_t maxBytes() const { return maxBytes_; }

  /// Update the cap. If new cap < current cachedBytes_, evicts LRU until
  /// under cap. Useful when caller knows memory budget at init.
  void setMaxBytes(size_t bytes);

  /// Pre-allocate buffers of these sizes and seed them into the cache so
  /// the first round of acquire() calls hit the cache instead of going to
  /// the device. Useful when the caller has a memory plan from AOTI.
  /// If total prewarmed bytes exceeds maxBytes_, oldest entries get evicted.
  void prewarm(const std::vector<size_t>& sizes);

private:
  void evictOldest();

  id<MTLDevice> device_;
  size_t maxBytes_;
  size_t cachedBytes_ = 0;

  struct PoolEntry {
    id<MTLBuffer> buffer;
    size_t size;
  };

  std::list<PoolEntry> lruList_;  // newest at front
  std::multimap<size_t, std::list<PoolEntry>::iterator> sizeMap_;

  static constexpr size_t kMaxHeadroom = 32768;  // 32KB
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
