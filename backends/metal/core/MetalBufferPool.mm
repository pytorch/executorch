/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt).
//
// Ownership: PoolEntry::buffer is __strong. Inserting an entry into
// lruList_ retains the buffer; eviction (evictOldest) and teardown
// (clear, ~MetalBufferPool) release via ARC when the entries are
// destroyed. The pool's strong ref is load-bearing for the cross-slot
// RAW fix (HazardTrackingTest.RAWHazardInsertsBarrier): the buffer
// can't dip below the pool's hold while the LRU list owns it.
//===----------------------------------------------------------------------===//

#import "MetalStream.h"
#include <executorch/runtime/platform/log.h>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MetalBufferPool
//===----------------------------------------------------------------------===//

MetalBufferPool::MetalBufferPool(id<MTLDevice> device, size_t maxBytes)
    : device_(device), maxBytes_(maxBytes) {}

MetalBufferPool::~MetalBufferPool() {
  clear();
}

id<MTLBuffer> MetalBufferPool::acquire(size_t size) {
  // Find best fit with bounded headroom
  auto it = sizeMap_.lower_bound(size);
  size_t doubleSize = (size > SIZE_MAX / 2) ? SIZE_MAX : 2 * size;
  size_t sizeWithHeadroom = (size > SIZE_MAX - kMaxHeadroom) ? SIZE_MAX : size + kMaxHeadroom;
  size_t maxAcceptable = std::min(doubleSize, sizeWithHeadroom);

  if (it != sizeMap_.end() && it->first <= maxAcceptable) {
    auto lruIt = it->second;
    id<MTLBuffer> buffer = lruIt->buffer;
    cachedBytes_ -= lruIt->size;
    lruList_.erase(lruIt);
    sizeMap_.erase(it);
    ET_LOG(Debug, "MetalBufferPool: reused %zu byte buffer (requested %zu)", [buffer length], size);
    return buffer;
  }

  id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
  ET_LOG(Debug, "MetalBufferPool: allocated new %zu byte buffer", size);
  return buffer;
}

void MetalBufferPool::release(id<MTLBuffer> buffer) {
  size_t size = [buffer length];

  // Don't pool very large buffers — drop on scope exit (anti-monopoly).
  if (size > maxBytes_ / 2) {
    return;
  }

  // PoolEntry's __strong buffer field retains; pool now holds the
  // load-bearing +1 (RAW-fix from HazardTrackingTest).
  lruList_.push_front({buffer, size});
  sizeMap_.insert({size, lruList_.begin()});
  cachedBytes_ += size;

  // Evict if over limit
  while (cachedBytes_ > maxBytes_ && !lruList_.empty()) {
    evictOldest();
  }
}

void MetalBufferPool::evictOldest() {
  auto tail = std::prev(lruList_.end());
  cachedBytes_ -= tail->size;

  // Find and remove from sizeMap
  auto range = sizeMap_.equal_range(tail->size);
  for (auto it = range.first; it != range.second; ++it) {
    if (it->second == tail) {
      sizeMap_.erase(it);
      break;
    }
  }

  lruList_.erase(tail);
}

void MetalBufferPool::clear() {
  // erase destroys all PoolEntry instances — each __strong buffer ref
  // is released.
  lruList_.clear();
  sizeMap_.clear();
  cachedBytes_ = 0;
}

void MetalBufferPool::setMaxBytes(size_t bytes) {
  maxBytes_ = bytes;
  // Shrink immediately if cached > new cap.
  while (cachedBytes_ > maxBytes_ && !lruList_.empty()) {
    evictOldest();
  }
}

void MetalBufferPool::prewarm(const std::vector<size_t>& sizes) {
  for (size_t size : sizes) {
    if (size == 0) continue;
    // Allocate the buffer and immediately seed it into the pool. Same
    // semantics as alloc(size) followed by free(ptr) — but skipping the
    // ptr → buffer round-trip (caller can't know GPU addr yet anyway).
    // Pool is residency-quiet: prewarm does NOT
    // pin to the residency set; per-CB binds drive residency on first
    // use. Honors maxBytes by evicting LRU first.
    while (cachedBytes_ + size > maxBytes_ && !lruList_.empty()) {
      evictOldest();
    }
    if (cachedBytes_ + size > maxBytes_) {
      // Single entry larger than cap — skip; user should bump capacity.
      ET_LOG(Info,
             "MetalBufferPool::prewarm: size %zu exceeds capacity %zu, skipping",
             size, maxBytes_);
      continue;
    }

    id<MTLBuffer> buffer =
        [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buffer) {
      ET_LOG(Error, "MetalBufferPool::prewarm: alloc(%zu) failed", size);
      continue;
    }
    lruList_.push_front({buffer, size});
    sizeMap_.insert({size, lruList_.begin()});
    cachedBytes_ += size;
  }
  ET_LOG(Info,
         "MetalBufferPool::prewarm: seeded %zu sizes (cached=%zu, cap=%zu)",
         sizes.size(), cachedBytes_, maxBytes_);
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
