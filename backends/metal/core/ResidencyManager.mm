/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). The
// `id set_` ivar (type-erased to avoid propagating MTLResidencySet's
// API_AVAILABLE annotations through the header) is __strong by default;
// ARC retains on assignment from -newResidencySetWithDescriptor: and
// releases on dtor. Locals like the descriptor are __strong scope-vars
// and ARC releases them at scope exit. This ivar is private to this TU
// so the ARC/MRC ABI ambiguity for shared id-typed struct fields does
// not apply.
//===----------------------------------------------------------------------===//

#import "ResidencyManager.h"

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

namespace {
// Identity-key extraction. The map does NOT extend buffer lifetime;
// caller (allocator/registry) holds the strong ref.
inline void* keyOf(id<MTLBuffer> buffer) {
  return (__bridge void*)buffer;
}
inline void* keyOf(id<MTLHeap> heap) {
  return (__bridge void*)heap;
}
} // namespace

ResidencyManager::ResidencyManager(id<MTLDevice> device) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
    desc.label = @"GpuStream ResidencySet";
    desc.initialCapacity = 64;

    NSError* error = nil;
    id<MTLResidencySet> set =
        [device newResidencySetWithDescriptor:desc error:&error];

    if (set) {
      set_ = set;
      enabled_ = true;
      // Mark the set as wants-to-be-resident exactly once. The driver's
      // "wants to be resident" disposition persists across subsequent
      // commits; per Apple docs requestResidency is a best-effort hint,
      // and MLX's production pattern (resident.cpp:24) calls it once at
      // construction. commit() then publishes individual add/remove
      // deltas without re-issuing the hint.
      [(id<MTLResidencySet>)set_ requestResidency];
      ET_LOG(Info, "ResidencyManager: Metal 4 ResidencySet enabled");
    } else {
      ET_LOG(Info, "ResidencyManager: ResidencySet not available: %s",
             error ? [[error localizedDescription] UTF8String] : "unknown");
    }
  }
#else
  (void)device;
#endif
}

ResidencyManager::~ResidencyManager() {
}

//===----------------------------------------------------------------------===//
// Mutex-acquisition helper. Increments mu_acquired_; tries try_lock first
// to detect contention.
//===----------------------------------------------------------------------===//

namespace {
struct ScopedLock {
  std::mutex& mu;
  std::atomic<uint64_t>& acquired;
  std::atomic<uint64_t>& contended;
  ScopedLock(std::mutex& m, std::atomic<uint64_t>& a, std::atomic<uint64_t>& c)
      : mu(m), acquired(a), contended(c) {
    if (!mu.try_lock()) {
      contended.fetch_add(1, std::memory_order_relaxed);
      mu.lock();
    }
    acquired.fetch_add(1, std::memory_order_relaxed);
  }
  ~ScopedLock() { mu.unlock(); }
  ScopedLock(const ScopedLock&) = delete;
  ScopedLock& operator=(const ScopedLock&) = delete;
};
} // namespace

//===----------------------------------------------------------------------===//
// add / remove route to refcounted pin/unpin so set membership is
// consistent regardless of which API a caller uses. The refcount layer
// adds a layer of bookkeeping on top. Caller must not mix add/remove
// with pin/unpin on the same buffer (would double-count the refcount).
//===----------------------------------------------------------------------===//

void ResidencyManager::add(id<MTLBuffer> buffer) {
  pin(buffer);
}
void ResidencyManager::remove(id<MTLBuffer> buffer) {
  unpin(buffer);
}

//===----------------------------------------------------------------------===//
// pin / unpin
//===----------------------------------------------------------------------===//

void ResidencyManager::pinLocked(id<MTLBuffer> buffer) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (!enabled_ || !set_ || !buffer) return;
    int& count = refcount_[keyOf(buffer)];
    ++count;
    if (count == 1) {
      [(id<MTLResidencySet>)set_ addAllocation:buffer];
      dirty_ = true;
    }
  }
#else
  (void)buffer;
#endif
}

void ResidencyManager::unpinLocked(id<MTLBuffer> buffer) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (!enabled_ || !set_ || !buffer) return;
    auto it = refcount_.find(keyOf(buffer));
    ET_CHECK_MSG(
        it != refcount_.end() && it->second > 0,
        "ResidencyManager::unpin: underflow on buffer %p (never pinned or "
        "already unpinned to zero)",
        keyOf(buffer));
    if (--it->second == 0) {
      refcount_.erase(it);
      [(id<MTLResidencySet>)set_ removeAllocation:buffer];
      dirty_ = true;
    }
  }
#else
  (void)buffer;
#endif
}

void ResidencyManager::pin(id<MTLBuffer> buffer) {
  pin_calls_.fetch_add(1, std::memory_order_relaxed);
  ScopedLock lk(mu_, mu_acquired_, mu_contended_);
  pinLocked(buffer);
}

void ResidencyManager::unpin(id<MTLBuffer> buffer) {
  unpin_calls_.fetch_add(1, std::memory_order_relaxed);
  ScopedLock lk(mu_, mu_acquired_, mu_contended_);
  unpinLocked(buffer);
}

void ResidencyManager::pinBatch(id<MTLBuffer> const __unsafe_unretained* buffers, size_t count) {
  pin_calls_.fetch_add(count, std::memory_order_relaxed);
  ScopedLock lk(mu_, mu_acquired_, mu_contended_);
  for (size_t i = 0; i < count; ++i) pinLocked(buffers[i]);
}

void ResidencyManager::unpinBatch(id<MTLBuffer> const __unsafe_unretained* buffers, size_t count) {
  unpin_calls_.fetch_add(count, std::memory_order_relaxed);
  ScopedLock lk(mu_, mu_acquired_, mu_contended_);
  for (size_t i = 0; i < count; ++i) unpinLocked(buffers[i]);
}

//===----------------------------------------------------------------------===//
// pinHeap / unpinHeap — one-shot, NOT refcounted (per design).
//===----------------------------------------------------------------------===//

void ResidencyManager::pinHeap(id<MTLHeap> heap) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (!heap) return;
    ScopedLock lk(mu_, mu_acquired_, mu_contended_);
    if (!enabled_ || !set_) return;
    [(id<MTLResidencySet>)set_ addAllocation:heap];
    dirty_ = true;
  }
#else
  (void)heap;
#endif
}

void ResidencyManager::unpinHeap(id<MTLHeap> heap) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (!heap) return;
    ScopedLock lk(mu_, mu_acquired_, mu_contended_);
    if (!enabled_ || !set_) return;
    [(id<MTLResidencySet>)set_ removeAllocation:heap];
    dirty_ = true;
  }
#else
  (void)heap;
#endif
}

//===----------------------------------------------------------------------===//
// commit / nudgeResidency
//===----------------------------------------------------------------------===//

void ResidencyManager::commit() {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    ScopedLock lk(mu_, mu_acquired_, mu_contended_);
    if (enabled_ && set_ && dirty_) {
      id<MTLResidencySet> rs = (id<MTLResidencySet>)set_;
      [rs commit];
      ET_LOG(Debug, "ResidencyManager: Committed (size=%llu bytes)",
             (unsigned long long)[rs allocatedSize]);
      dirty_ = false;
    }
  }
#endif
}

void ResidencyManager::nudgeResidency() {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    ScopedLock lk(mu_, mu_acquired_, mu_contended_);
    if (enabled_ && set_) {
      [(id<MTLResidencySet>)set_ requestResidency];
    }
  }
#endif
}

//===----------------------------------------------------------------------===//
// Queue wiring.
//===----------------------------------------------------------------------===//

void ResidencyManager::addQueueResidency(id<MTLCommandQueue> queue) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (enabled_ && set_ && queue) {
      [queue addResidencySet:(id<MTLResidencySet>)set_];
    }
  }
#else
  (void)queue;
#endif
}

void ResidencyManager::removeQueueResidency(id<MTLCommandQueue> queue) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (enabled_ && set_ && queue) {
      [queue removeResidencySet:(id<MTLResidencySet>)set_];
    }
  }
#else
  (void)queue;
#endif
}

#if ET_METAL4_AVAILABLE
void ResidencyManager::addQueueResidency(id<MTL4CommandQueue> queue) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (enabled_ && set_ && queue) {
      [queue addResidencySet:(id<MTLResidencySet>)set_];
    }
  }
}

void ResidencyManager::removeQueueResidency(id<MTL4CommandQueue> queue) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (enabled_ && set_ && queue) {
      [queue removeResidencySet:(id<MTLResidencySet>)set_];
    }
  }
}
#endif

//===----------------------------------------------------------------------===//
// Debug accessors.
//===----------------------------------------------------------------------===//

int ResidencyManager::refcountForTesting(id<MTLBuffer> buffer) const {
  if (!buffer) return 0;
  std::lock_guard<std::mutex> lk(mu_);
  auto it = refcount_.find(keyOf(buffer));
  return it == refcount_.end() ? 0 : it->second;
}

ResidencyManager::Stats ResidencyManager::stats() const {
  return Stats{
      pin_calls_.load(std::memory_order_relaxed),
      unpin_calls_.load(std::memory_order_relaxed),
      mu_acquired_.load(std::memory_order_relaxed),
      mu_contended_.load(std::memory_order_relaxed),
  };
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
