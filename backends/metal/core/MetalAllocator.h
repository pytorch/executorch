/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalAllocator — owns all per-stream memory state.
// Owns:
//   - BufferRegistry      : host-ptr → (MTLBuffer, offset, origin) mapping
//   - MetalBufferPool     : LRU recycling cache for alloc()/free()
//   - MetalHeap           : optional pre-allocated heap arena
//   - ResidencyManager    : MTL4 GPU-residency set
// Borrows (non-owning):
//   - id<MTLDevice>       : for buffer creation
//   - HazardTracker*      : ONE method (notifyExternalWrite) needs to record
//                           a host-side write into the hazard tracker. This
//                           is the one cross-class back-edge from Allocator
//                           into the Recorder; an explicit borrow keeps the
//                           dependency obvious.
// Lifetime: per-stream (one MetalAllocator per MetalStream instance).
//===----------------------------------------------------------------------===//

#include <executorch/backends/metal/core/BufferRegistry.h>
#include <executorch/backends/metal/core/MetalBufferPool.h>
#include <executorch/backends/metal/core/MetalHeap.h>
#include <executorch/backends/metal/core/ResidencyManager.h>

#import <Metal/Metal.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

class HazardTracker;

class MetalAllocator {
 public:
  // ctor: device + hazards (borrowed). hazards may be null only in tests
  // that don't exercise notifyExternalWrite.
  MetalAllocator(id<MTLDevice> device, HazardTracker* hazards);
  ~MetalAllocator();
  MetalAllocator(const MetalAllocator&) = delete;
  MetalAllocator& operator=(const MetalAllocator&) = delete;

  // {parent_mtl, byte_offset_within_parent}. Same shape as the binding
  // information setBuffer:offset:atIndex: needs from Metal.
  struct BufferBinding {
    id<MTLBuffer> mtl;
    size_t offset;
  };

  //===--------------------------------------------------------------------===//
  // alloc / free
  //===--------------------------------------------------------------------===//

  // Allocate `bytes` of GPU-accessible memory. Tries heap first (if
  // enabled), falls back to the LRU pool, falls back to a fresh device
  // alloc. Returns the host-addressable pointer (Apple Silicon unified
  // memory).
  void* alloc(size_t bytes);

  // Free a pointer returned by alloc() (or a registered external buffer).
  // Routes by Origin: Pool → return to LRU; Heap → drop wrapper; External*
  // → remove from residency set + drop wrapper; Subregion → no-op
  // (parent owns the MTLBuffer).
  void free(void* ptr);

  //===--------------------------------------------------------------------===//
  // External buffer registration
  //===--------------------------------------------------------------------===//

  // Re-export for ergonomics. Same enum as BufferRegistry::ResidencyClass.
  using ResidencyClass = BufferRegistry::ResidencyClass;

  // Wrap an existing host pointer in an MTLBuffer (zero-copy if page-
  // aligned, else a new buffer with copied bytes). strict_zero_copy=true
  // refuses the copy fallback. Returns false on failure.
  //
  // Residency lifetime:
  //   ResidencyClass::Transient (default) — no allocator-side pin at
  //     register; per-CB binds via the recorder pin/unpin around in-flight
  //     CBs. Suitable for typical AOTI inputs/outputs.
  //   ResidencyClass::Permanent — pin into the residency set at register
  //     time; stays resident until the caller explicitly invokes
  //     unregisterExternalBuffer(ptr). Suitable for long-lived
  //     model weights / scratch that the caller wants to keep resident
  //     across many CBs without per-CB churn.
  //
  // Cleanup: call unregisterExternalBuffer(ptr) — works for both
  // residency classes (routes internally). free(ptr) also accepts
  // External entries for backward compat / auto-registered entries
  // (see bufferForPtr).
  bool registerExternalBuffer(
      void* ptr,
      size_t bytes,
      bool strict_zero_copy = false,
      ResidencyClass residency_class = ResidencyClass::Transient);

  // Unregister an entry previously registered via registerExternalBuffer
  // (or auto-registered via bufferForPtr). Routes by ResidencyClass:
  //   Transient — drops the MTLBuffer wrapper; nothing to unpin (transient
  //               externals never had an alloc-time pin).
  //   Permanent — drops the long-lived residency pin AND removes the
  //               registry entry. Caller MUST ensure NO in-flight CB on
  //               ANY stream still binds `ptr` (or any subregion of it)
  //               before calling — otherwise the GPU may dereference a
  //               buffer that's been removed from the residency set.
  //               Debug builds DCHECK that the global residency refcount
  //               equals exactly 1 at unregister time (the registration's
  //               own pin is the only expected contributor).
  //
  // FATAL on caller bug:
  //   - `ptr` is null or not registered (call site has a logic error).
  //   - `ptr` is registered but with a non-External Origin (e.g., a Pool
  //     or Subregion entry — those have their own cleanup paths).
  //
  // The caller's underlying memory is NOT touched — only the MTLBuffer
  // wrapper and (for Permanent) the residency pin are released.
  void unregisterExternalBuffer(void* ptr);

  //===--------------------------------------------------------------------===//
  // Subregion API — child_ptr is registered as a sub-buffer view of an
  // already-registered parent_ptr. Used by AOTI's _reinterpret_tensor.
  //===--------------------------------------------------------------------===//

  bool registerSubregion(
      void* child_ptr,
      void* parent_ptr,
      size_t offset,
      size_t size);
  void unregisterSubregion(void* child_ptr);

  //===--------------------------------------------------------------------===//
  // Address resolution — resolve a host pointer to its MTLBuffer.
  //===--------------------------------------------------------------------===//

  // Resolve `ptr` to (parent_mtl, byte_offset). Auto-registers the host
  // memory region if not yet known. Returns {nil, 0} on registration
  // failure (caller should fall back to inline-copy via setBytes).
  BufferBinding bufferForPtr(void* ptr, size_t bytes);

  // Like bufferForPtr but returns just the MTLBuffer (no offset). Used
  // by callers that don't care about offsets (e.g., MPSGraphTensorData
  // wrappers that want the parent buffer).
  // ⚠️ For Subregion entries, this returns the *parent* MTLBuffer with
  // no offset information. Callers that need the offset MUST use
  // bufferForPtr() instead.
  id<MTLBuffer> bufferMtlForPtr(void* ptr, size_t bytes);

  // Direct registry probe (no auto-register). Returns the entry pointer
  // or nullptr on miss. Used by callers that need to inspect Origin or
  // size without forcing registration.
  const BufferRegistry::Entry* findEntry(void* ptr);

  //===--------------------------------------------------------------------===//
  // Hazard interop — host-side write notification.
  //===--------------------------------------------------------------------===//

  // Notify that the host wrote to `ptr` (e.g. via memcpy). Resolves to
  // the underlying MTLBuffer and records the range in the hazard tracker
  // so the next GPU dispatch reading any overlapping range pre-barriers.
  // No-op if `ptr` is not registered.
  void notifyExternalWrite(void* ptr, size_t bytes);

  //===--------------------------------------------------------------------===//
  // Heap configuration (optional fast-path arena)
  //===--------------------------------------------------------------------===//

  void enableHeap(size_t heapSizeBytes, bool aliasable = false);
  bool heapEnabled() const {
    return useHeap_;
  }

  //===--------------------------------------------------------------------===//
  // Pool tuning
  //===--------------------------------------------------------------------===//

  void setPoolCapacity(size_t bytes) {
    if (pool_)
      pool_->setMaxBytes(bytes);
  }
  void prewarm(const std::vector<size_t>& sizes) {
    if (pool_)
      pool_->prewarm(sizes);
  }

  //===--------------------------------------------------------------------===//
  // ResidencyManager access (borrowed). MetalMTL4Backend needs this at
  // its own construction time so it can register its scratch buffer in
  // the residency set.
  //
  // Two accessors:
  //   - residency()       : raw pointer (borrowed; for hot paths like
  //                          recorder.flush()'s pinBatch).
  //   - residencyShared() : shared_ptr (for capturing into completion
  //                          handlers that may outlive this allocator —
  //                          per-CB unpin path;
  //                          the shared ref keeps the manager alive until
  //                          the last handler fires).
  //===--------------------------------------------------------------------===//

  ResidencyManager* residency() {
    return residency_.get();
  }
  std::shared_ptr<ResidencyManager> residencyShared() {
    return residency_;
  }

 private:
  id<MTLDevice> device_; // borrowed
  HazardTracker* hazards_; // borrowed; may be nullptr in tests

  // Owned subsystems.
  BufferRegistry buffers_;
  std::unique_ptr<MetalBufferPool> pool_;
  std::unique_ptr<MetalHeap> heap_;
  bool useHeap_ = false;
  // shared_ptr instead of unique_ptr so completion handlers can capture
  // a strong ref that outlives this allocator's destruction window
  // (per-CB unpin path — see the residencyShared() comment above).
  std::shared_ptr<ResidencyManager> residency_;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
