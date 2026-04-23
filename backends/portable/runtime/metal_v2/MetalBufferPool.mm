/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
    : device_(device), maxBytes_(maxBytes) {
  [device_ retain];
}

MetalBufferPool::~MetalBufferPool() {
  clear();
  [device_ release];
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

  // Allocate new
  id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
  ET_LOG(Debug, "MetalBufferPool: allocated new %zu byte buffer", size);
  return buffer;
}

void MetalBufferPool::release(id<MTLBuffer> buffer) {
  size_t size = [buffer length];

  // Don't pool very large buffers
  if (size > maxBytes_ / 2) {
    [buffer release];
    return;
  }

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

  [tail->buffer release];
  lruList_.erase(tail);
}

void MetalBufferPool::clear() {
  for (auto& entry : lruList_) {
    [entry.buffer release];
  }
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
    // ptr → buffer round-trip and the residency-set add (caller can't
    // know GPU addr yet anyway). Honors maxBytes by evicting LRU first.
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
      ET_LOG(Error, "MetalBufferPool::prewarm: alloc of %zu failed", size);
      continue;
    }
    PoolEntry entry{buffer, size};
    lruList_.push_front(entry);
    sizeMap_.insert({size, lruList_.begin()});
    cachedBytes_ += size;
  }
  ET_LOG(Info,
         "MetalBufferPool::prewarm: seeded %zu buffers, cached=%zu/%zu bytes",
         sizes.size(), cachedBytes_, maxBytes_);
}


} // namespace metal_v2
} // namespace backends
} // namespace executorch

