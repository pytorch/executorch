/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ResidencyManager — wraps an MTLResidencySet (macOS 15+ / iOS 18+).
//
// Two API tiers:
//
//   Set-semantic add/remove (kept for callers that don't need refcounting):
//     add / remove   — Callers must NOT mix add/remove with the
//                      refcounted pin/unpin on the same buffer.
//
//   Refcounted:
//     pin / unpin             — refcounted; only 0↔1 transitions touch
//                               Metal's set; flips dirty_ on transition.
//     pinBatch / unpinBatch   — single mutex acquisition for N entries;
//                               caller must dedup against bound_buffers_
//                               before calling (input must be
//                               duplicate-free).
//     pinHeap / unpinHeap     — one-shot pin of an MTLHeap that conforms
//                               to MTLAllocation. Single ref (NOT
//                               refcounted). Pairs with unpinHeap. If
//                               multiple heaps are ever supported, must
//                               be promoted to refcounted.
//     commit()                — flush pending deltas via [set commit].
//                               requestResidency runs ONCE in the ctor;
//                               not repeated here.
//     nudgeResidency()        — re-issues [set requestResidency]. Manual
//                               escape hatch for memory-pressure
//                               eviction recovery; not called in steady
//                               state.
//     addQueueResidency /
//     removeQueueResidency    — attach/detach the set on a command queue
//                               via the standard Metal APIs. Encapsulates
//                               the set access so callers don't need a
//                               raw set handle.
//
//   Debug-only:
//     refcountForTesting(buf) — current refcount (test-only accessor).
//     stats()                 — snapshot of pin/unpin/mutex counters.
//
// Underflow is fatal (`ET_CHECK_MSG`).
//
// Thread-safe: a single internal mutex serializes pin/unpin/pinBatch/
// unpinBatch/commit/nudgeResidency. The set's "wants to be resident"
// disposition is set ONCE at ctor; commits publish add/remove deltas.
//
// Lifecycle
// ---------
// - Constructor: -newResidencySetWithDescriptor: (gated on @available
//   + ET_METAL4_AVAILABLE), then -requestResidency once. On older OSes
//   the manager is constructed but isEnabled() returns false and all
//   calls are no-ops.
// - Destructor releases the set.

#import <Metal/Metal.h>

#if !defined(ET_METAL4_AVAILABLE)
#include <executorch/backends/metal/core/MetalConfig.h>
#endif

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

class ResidencyManager {
 public:
  // device is borrowed (caller owns lifetime).
  explicit ResidencyManager(id<MTLDevice> device);
  ~ResidencyManager();

  ResidencyManager(const ResidencyManager&) = delete;
  ResidencyManager& operator=(const ResidencyManager&) = delete;

  //===--------------------------------------------------------------------===//
  // Set-semantic add/remove. Do not mix with the refcounted pin/unpin
  // on the same buffer.
  //===--------------------------------------------------------------------===//
  void add(id<MTLBuffer> buffer);
  void remove(id<MTLBuffer> buffer);

  //===--------------------------------------------------------------------===//
  // Refcounted API.
  //===--------------------------------------------------------------------===//

  // Refcount +1; on 0→1 transition: [set addAllocation:buf], set dirty_.
  // Thread-safe.
  void pin(id<MTLBuffer> buffer);

  // Refcount -1; on 1→0 transition: [set removeAllocation:buf], erase,
  // set dirty_. Underflow on never-pinned is fatal (ET_CHECK_MSG).
  // Thread-safe.
  void unpin(id<MTLBuffer> buffer);

  // Batched variants. Single mutex acquisition for N entries. Caller
  // MUST dedup against its own per-CB bound_buffers_ set before
  // calling; input must be duplicate-free.
  //
  // Public API takes raw (pointer, count) instead of std::span to
  // avoid imposing C++20 on every header consumer.
  //
  // The `__unsafe_unretained` qualifier on id<MTLBuffer>* makes the
  // function signature mangle the same way under both ARC and MRC
  // compilation contexts (default ARC qualifier is __autoreleasing,
  // default MRC is __unsafe_unretained — they produce different mangled
  // C++ symbol names → linker mismatch when one TU is ARC and another
  // is MRC). Pinning to __unsafe_unretained makes both sides agree.
  void pinBatch(id<MTLBuffer> const __unsafe_unretained* buffers, size_t count);
  void unpinBatch(id<MTLBuffer> const __unsafe_unretained* buffers, size_t count);

  // One-shot pin/unpin of an MTLHeap. NOT refcounted. Pair them.
  void pinHeap(id<MTLHeap> heap);
  void unpinHeap(id<MTLHeap> heap);

  //===--------------------------------------------------------------------===//
  // Commit / residency hint.
  //===--------------------------------------------------------------------===//

  // Flush pending add/remove deltas via [set commit]. requestResidency
  // is amortized to a single ctor call; not repeated here.
  void commit();

  // Re-issue [set requestResidency]. Manual escape hatch for memory-
  // pressure-eviction recovery; not called by the runtime in steady
  // state.
  void nudgeResidency();

  //===--------------------------------------------------------------------===//
  // Queue wiring. Encapsulates the set access; callers shouldn't need a
  // raw set handle.
  //===--------------------------------------------------------------------===//

  void addQueueResidency(id<MTLCommandQueue> queue);
  void removeQueueResidency(id<MTLCommandQueue> queue);
#if ET_METAL4_AVAILABLE
  // MTL4 queue uses a different protocol (id<MTL4CommandQueue>).
  // Overload is gated on availability.
  void addQueueResidency(id<MTL4CommandQueue> queue) API_AVAILABLE(macos(26.0), ios(26.0));
  void removeQueueResidency(id<MTL4CommandQueue> queue) API_AVAILABLE(macos(26.0), ios(26.0));
#endif

  bool isEnabled() const { return enabled_; }

  //===--------------------------------------------------------------------===//
  // Debug accessors. NOT for production code paths.
  //===--------------------------------------------------------------------===//

  // Current refcount for `buffer` in the persistent set. 0 if not
  // pinned. Thread-safe (acquires mu_).
  int refcountForTesting(id<MTLBuffer> buffer) const;

  // Counters for observability. Each member is a snapshot of the
  // corresponding atomic at the time of the call. Not atomic-as-a-
  // group; intended for low-frequency monitoring (e.g., periodic
  // ods/scuba reporter).
  struct Stats {
    uint64_t pin_calls;
    uint64_t unpin_calls;
    uint64_t mu_acquired;
    uint64_t mu_contended;
  };
  Stats stats() const;

  //===--------------------------------------------------------------------===//
  // Direct accessor for the underlying MTLResidencySet. New code should
  // prefer addQueueResidency / removeQueueResidency. Returns nil when
  // disabled.
  //===--------------------------------------------------------------------===//
#if ET_METAL4_AVAILABLE
  id<MTLResidencySet> nativeSet() const API_AVAILABLE(macos(15.0), ios(18.0)) {
    return set_;
  }
#endif

 private:
  // Single mutex for refcount map mutations + dirty_ flag + Metal set
  // mutating calls (addAllocation/removeAllocation/commit). The Metal
  // residency set's mutating API is not documented as thread-safe per-
  // set; we serialize.
  mutable std::mutex mu_;

  // Per-MTLBuffer refcount, identity-keyed by (__bridge void*)id. Map
  // does NOT extend buffer lifetime; the caller (allocator/registry)
  // holds the strong ref.
  std::unordered_map<void*, int> refcount_;

  // Set is gated on @available macOS 15+ / iOS 18+. When the gate
  // doesn't pass we leave set_ as nil and enabled_ as false; all calls
  // become no-ops. Stored as id (untyped) so the field declaration
  // doesn't need API_AVAILABLE — see "type-erasure"
  // note. Cast to id<MTLResidencySet> at use site under @available.
  id set_ = nil;
  bool enabled_ = false;
  // Tracks whether 0↔1 refcount transitions have changed the set since
  // the last commit. commit() is a no-op when false.
  bool dirty_ = false;

  // Counters for observability. Atomic so reads in stats() don't need
  // the mutex.
  std::atomic<uint64_t> pin_calls_{0};
  std::atomic<uint64_t> unpin_calls_{0};
  std::atomic<uint64_t> mu_acquired_{0};
  std::atomic<uint64_t> mu_contended_{0};

  // Internal helpers (caller must hold mu_).
  void pinLocked(id<MTLBuffer> buffer);
  void unpinLocked(id<MTLBuffer> buffer);
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
