/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). The
// allocator goes hand in hand with BufferRegistry (also ARC); Entry::mtl
// is __strong, so the registry's insert / remove / clear retain and
// release automatically.
//===----------------------------------------------------------------------===//

#import "MetalAllocator.h"
#import "HazardTracker.h"

#include <executorch/runtime/platform/log.h>

#include <unistd.h>  // sysconf, _SC_PAGESIZE

namespace executorch {
namespace backends {
namespace metal_v2 {

MetalAllocator::MetalAllocator(id<MTLDevice> device, HazardTracker* hazards)
    : device_(device), hazards_(hazards) {
  pool_ = std::make_unique<MetalBufferPool>(device_);
  // Metal 4: ResidencySet for GPU-resident memory. The manager handles
  // OS-availability gating internally; if the API isn't available it's
  // a no-op. Owned via shared_ptr so completion handlers can capture
  // a strong ref (see MetalAllocator.h's residencyShared()).
  residency_ = std::make_shared<ResidencyManager>(device_);
}

MetalAllocator::~MetalAllocator() {
  // Cross-class ordering invariant:
  //   Caller MUST destroy MetalCommandRecorder before MetalAllocator.
  // The recorder's destructor calls wait(), which drains pending
  // completion handlers (each handler captures a shared_ptr to
  // residency_ and calls unpinBatch on it). Destroying the allocator
  // first would either UAF (if the handler still holds a raw ptr) or
  // leak the residency entries (if the shared_ptr extends lifetime).
  // MetalStream enforces the order by the field declaration sequence;
  // standalone consumers must replicate it.
  //
  // Note: walk the registry surfacing any
  // Permanent External entries the caller forgot to unregister. We do
  // NOT defensively unpin those entries — the dtor lacks the
  // information to know whether other streams still have in-flight CBs
  // that bind them, and a defensive unpin could race with a CB-
  // completion handler's unpinBatch on the same buffer → refcount
  // underflow → fatal ET_CHECK_MSG. The "safety net" turns into a
  // process abort. Letting the residency-set entry live to process
  // exit is a bounded leak (caller bug, not infinite growth); the OS
  // reclaims everything on exit. The loud Error log surfaces the
  // caller bug for fixing without taking down the process.
  //
  // After this walk, ~BufferRegistry handles the wrapper-release pass
  // for all entries (Pool / Heap / External*); pool_ destructor handles
  // the LRU cache. The walk itself is read-only on the registry.
  buffers_.forEach([](void* ptr, const BufferRegistry::Entry& entry) {
    const bool is_external =
        entry.origin == BufferRegistry::Origin::ExternalAliased ||
        entry.origin == BufferRegistry::Origin::ExternalCopied;
    if (is_external &&
        entry.residency_class ==
            BufferRegistry::ResidencyClass::Permanent) {
      ET_LOG(Error,
             "MetalAllocator dtor: Permanent External entry for ptr %p "
             "leaked (caller forgot unregisterExternalPermanent); "
             "residency-set entry retained until process exit",
             ptr);
    }
  });
}

//===----------------------------------------------------------------------===//
// alloc / free
//===----------------------------------------------------------------------===//

void* MetalAllocator::alloc(size_t bytes) {
  id<MTLBuffer> buffer = nil;
  bool from_heap = false;

  // Try heap first (faster: ~100ns vs ~10µs)
  if (useHeap_ && heap_) {
    buffer = heap_->allocBuffer(bytes);
    if (buffer) from_heap = true;
  }

  // Fallback to buffer pool
  if (!buffer) {
    buffer = pool_->acquire(bytes);
  }

  if (!buffer) {
    ET_LOG(Error, "MetalAllocator::alloc: failed to allocate %zu bytes", bytes);
    return nullptr;
  }

  void* ptr = [buffer contents];
  // Tag heap-allocated buffers as Origin::Heap so free() routes them to
  // the heap (a no-op release of the wrapper) instead of the pool (which
  // would cache them as if they came from newBufferWithLength and corrupt
  // the size buckets).
  const auto origin = from_heap
      ? BufferRegistry::Origin::Heap
      : BufferRegistry::Origin::Pool;
  buffers_.insert(ptr, buffer, origin, bytes);

  // Note: no alloc-time pin. Pool/Heap
  // origin buffers enter the residency set via per-CB binds at flush()
  // time (MetalCommandRecorder::pinBatch) and leave at wait() time
  // (unpinBatch).
  //
  // External buffers still get pinned at registerExternalBuffer and
  // unpinned at free's External branch — their lifecycle is bounded by
  // explicit register/unregister pairing.

  ET_LOG(Debug, "MetalAllocator::alloc: %zu bytes at %p (heap=%d)",
         bytes, ptr, (int)from_heap);
  return ptr;
}

void MetalAllocator::free(void* ptr) {
  if (!ptr) return;
  auto removed = buffers_.remove(ptr);
  if (!removed) return;
  // Routing by Origin (see header):
  //   Subregion       — borrows parent's mtl; nothing to do here
  //   Pool            — return to pool; leave in residency set
  //   Heap            — heap arena is bump-allocated; just drop the
  //                     wrapper retain. Leave in residency set.
  //   External*       — caller's done with it for good; remove from
  //                     residency set, drop wrapper.
  if (removed->origin == BufferRegistry::Origin::Subregion) {
    return;
  }
  if (removed->origin == BufferRegistry::Origin::Pool) {
    pool_->release(removed->mtl);
  } else if (removed->origin == BufferRegistry::Origin::Heap) {
    // No per-buffer return path for heap arena; only drop the wrapper.
  } else {
    // External*:
    // Transient externals: never alloc-time-pinned (per-CB binds covered
    //   them); free() drops the wrapper, no residency call needed.
    // Permanent externals: alloc-time-pinned. The proper unpin path is
    //   unregisterExternalPermanent(ptr), which calls unpin AND removes
    //   from the registry. If the caller went through free() directly
    //   without first unregistering, the residency entry leaks until
    //   process exit (~ResidencyManager teardown). Log a warning so the
    //   caller bug surfaces; do NOT call unpin here — the dtor lacks
    //   the information to know whether other streams still have
    //   in-flight CBs binding `removed->mtl` (defensive unpin would race
    //   with completion-handler unpinBatch and underflow the refcount).
    if (removed->residency_class == BufferRegistry::ResidencyClass::Permanent) {
      ET_LOG(
          Error,
          "MetalAllocator::free: Permanent External entry for ptr %p "
          "freed without prior unregisterExternalPermanent(); residency "
          "entry leaks until process exit",
          ptr);
    }
  }
}

//===----------------------------------------------------------------------===//
// External buffer registration
//===----------------------------------------------------------------------===//

bool MetalAllocator::registerExternalBuffer(
    void* ptr, size_t bytes, bool strict_zero_copy,
    ResidencyClass residency_class) {
  if (!ptr || bytes == 0) return false;

  // Cache hit: refresh the snapshot for ExternalCopied entries.
  if (auto* entry = buffers_.find(ptr)) {
    // Re-registration must request the same residency class as the
    // existing entry. A Transient → Permanent (or vice-versa) silent
    // upgrade would skip the residency_->pin call below; a later
    // unregisterExternalPermanent(ptr) would then attempt to unpin a
    // buffer that was never pinned → fatal underflow in
    // ResidencyManager::unpinLocked. Better to fail loudly at the
    // mismatched-register call site than crash later.
    ET_DCHECK_MSG(
        entry->residency_class == residency_class,
        "MetalAllocator::registerExternalBuffer: ptr=%p already registered "
        "as %s; cannot re-register as %s. Unregister first if you need to "
        "change the residency class.",
        ptr,
        entry->residency_class == BufferRegistry::ResidencyClass::Permanent
            ? "Permanent" : "Transient",
        residency_class == BufferRegistry::ResidencyClass::Permanent
            ? "Permanent" : "Transient");
    buffers_.refreshIfCopied(ptr, bytes);
    return true;
  }

  // Page alignment check (typically 16 KB on Apple silicon, 4 KB under
  // Rosetta). sysconf is portable and survives future page-size changes.
  bool pageAligned = ((uintptr_t)ptr % sysconf(_SC_PAGESIZE)) == 0;
  ET_LOG(Debug,
         "MetalAllocator: registering external %p (%zu bytes, page_aligned=%d, strict=%d)",
         ptr, bytes, pageAligned, strict_zero_copy);

  // Try zero-copy first.
  id<MTLBuffer> buffer = [device_ newBufferWithBytesNoCopy:ptr
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
  BufferRegistry::Origin origin = BufferRegistry::Origin::ExternalAliased;

  if (!buffer) {
    if (strict_zero_copy) {
      ET_LOG(Info,
             "MetalAllocator: zero-copy wrap failed for %p (%zu bytes); strict mode -> refusing fallback",
             ptr, bytes);
      return false;
    }
    // Fallback: copy bytes into a fresh buffer.
    ET_LOG(Debug, "MetalAllocator: noCopy failed (alignment?), falling back to copy");
    buffer = [device_ newBufferWithBytes:ptr
                                  length:bytes
                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
      ET_LOG(Error,
             "MetalAllocator: failed to create buffer for external memory %p", ptr);
      return false;
    }
    origin = BufferRegistry::Origin::ExternalCopied;
  }

  buffers_.insert(ptr, buffer, origin, bytes, residency_class);

  // Note: only Permanent externals get an
  // alloc-time pin. Transient externals enter the residency set lazily
  // via per-CB binds at recorder commit.
  if (residency_ &&
      residency_class == ResidencyClass::Permanent) {
    residency_->pin(buffer);
  }

  ET_LOG(Debug, "MetalAllocator: registered %p -> MTLBuffer %p (origin=%d, class=%d)",
         ptr, (__bridge void*)buffer, (int)origin, (int)residency_class);
  return true;
}

void MetalAllocator::unregisterExternalPermanent(void* ptr) {
  const auto* entry = buffers_.find(ptr);
  if (!entry) {
    // Not registered — idempotent no-op.
    return;
  }
  if (entry->residency_class != BufferRegistry::ResidencyClass::Permanent) {
    ET_LOG(Error,
           "MetalAllocator::unregisterExternalPermanent: ptr %p was "
           "registered as Transient, not Permanent; skipping (caller bug)",
           ptr);
    return;
  }

  id<MTLBuffer> buf = entry->mtl;

  // Debug-mode enforcement: refcount must equal
  // exactly 1 — the registration's own pin. Refcount > 1 means an
  // in-flight CB on some stream still references this buffer; calling
  // unpin now would yank it out of the residency set during in-flight
  // GPU work → undefined behavior on MTL4. Caller MUST sync all streams
  // that may have bound `ptr` before unregistering.
#if !defined(NDEBUG)
  if (residency_) {
    int rc = residency_->refcountForTesting(buf);
    ET_DCHECK_MSG(
        rc == 1,
        "unregisterExternalPermanent: ptr=%p refcount=%d (expected 1). "
        "In-flight CB still binds this buffer; sync all streams first.",
        ptr, rc);
  }
#endif

  if (residency_) residency_->unpin(buf);

  // Drop from the registry. This mirrors the External-branch logic in
  // free() but skips the leak warning since we did the proper unpin.
  (void)buffers_.remove(ptr);
  ET_LOG(Debug, "MetalAllocator::unregisterExternalPermanent: %p", ptr);
}

//===----------------------------------------------------------------------===//
// Subregion API
//===----------------------------------------------------------------------===//

bool MetalAllocator::registerSubregion(
    void* child_ptr, void* parent_ptr, size_t offset, size_t size) {
  if (!child_ptr || !parent_ptr || size == 0) return false;

  // Idempotent: same binding → no-op.
  if (auto* existing = buffers_.find(child_ptr)) {
    if (existing->origin == BufferRegistry::Origin::Subregion &&
        existing->offset == offset &&
        existing->size == size) {
      return true;
    }
    ET_LOG(Error,
           "MetalAllocator::registerSubregion: child_ptr %p already registered "
           "with different binding (existing origin=%d offset=%zu size=%zu; "
           "new offset=%zu size=%zu). Call unregisterSubregion first.",
           child_ptr, (int)existing->origin, existing->offset, existing->size,
           offset, size);
    return false;
  }

  // Resolve parent and walk up if it's itself a subregion.
  const BufferRegistry::Entry* parent_entry = buffers_.find(parent_ptr);
  if (!parent_entry || !parent_entry->mtl) {
    ET_LOG(Error,
           "MetalAllocator::registerSubregion: parent_ptr %p is not registered "
           "(child_ptr=%p, offset=%zu, size=%zu).",
           parent_ptr, child_ptr, offset, size);
    return false;
  }
  id<MTLBuffer> root_mtl = parent_entry->mtl;
  size_t root_offset = parent_entry->offset + offset;
  buffers_.insertSubregion(child_ptr, root_mtl, root_offset, size);
  return true;
}

void MetalAllocator::unregisterSubregion(void* child_ptr) {
  if (!child_ptr) return;
  const BufferRegistry::Entry* entry = buffers_.find(child_ptr);
  if (!entry || entry->origin != BufferRegistry::Origin::Subregion) return;
  // Subregion entries borrow the parent's MTLBuffer ref; remove() returns
  // the entry but we do NOT release entry.mtl.
  (void)buffers_.remove(child_ptr);
}

//===----------------------------------------------------------------------===//
// Address resolution
//===----------------------------------------------------------------------===//

MetalAllocator::BufferBinding MetalAllocator::bufferForPtr(
    void* ptr, size_t bytes) {
  // Single hashmap lookup on warm path. (Was contains+find = 2 hashes
  // on the same key — costly with libstdc++ unordered_map<void*, …>.)
  auto* entry = buffers_.find(ptr);
  if (!entry) {
    registerExternalBuffer(ptr, bytes);
    entry = buffers_.find(ptr);
    if (!entry) return {nil, 0};
  } else if (entry->origin == BufferRegistry::Origin::ExternalCopied) {
    // Hot path for ExternalCopied: refresh the snapshot every time so
    // mutated host data (stack-local std::vector etc.) is visible to
    // the GPU. registerExternalBuffer's first-touch path does this too.
    buffers_.refreshIfCopied(ptr, bytes);
  }
  // Size validation: caller must not ask for more bytes than the entry
  // covers. For Subregion entries the size is the SUBREGION size (not
  // the parent's), so requesting more would silently read past the
  // intended view into adjacent memory.
  ET_DCHECK_MSG(
      bytes <= entry->size,
      "MetalAllocator::bufferForPtr: ptr=%p requested %zu bytes but the "
      "registered entry only covers %zu bytes (origin=%d). Re-register "
      "the larger region or use a different ptr.",
      ptr, bytes, entry->size, static_cast<int>(entry->origin));
  size_t offset = (entry->origin == BufferRegistry::Origin::Subregion)
      ? entry->offset
      : 0;
  return {entry->mtl, offset};
}

id<MTLBuffer> MetalAllocator::bufferMtlForPtr(void* ptr, size_t bytes) {
  return bufferForPtr(ptr, bytes).mtl;
}

const BufferRegistry::Entry* MetalAllocator::findEntry(void* ptr) {
  return buffers_.find(ptr);
}

//===----------------------------------------------------------------------===//
// Hazard interop
//===----------------------------------------------------------------------===//

void MetalAllocator::notifyExternalWrite(void* ptr, size_t size) {
  if (!ptr || size == 0 || !hazards_) return;
  const BufferRegistry::Entry* entry = buffers_.find(ptr);
  if (!entry || !entry->mtl) {
    // Not tracked — best-effort; nothing to do.
    return;
  }
  id<MTLBuffer> mtl = entry->mtl;
  size_t offset = (entry->origin == BufferRegistry::Origin::Subregion)
      ? entry->offset
      : 0;
  hazards_->notifyExternalWrite(mtl, offset, offset + size);
}

//===----------------------------------------------------------------------===//
// Heap configuration
//===----------------------------------------------------------------------===//

void MetalAllocator::enableHeap(size_t heapSizeBytes, bool aliasable) {
  if (heap_) {
    ET_LOG(Info, "MetalAllocator: heap already enabled");
    return;
  }
  heap_ = std::make_unique<MetalHeap>(device_, heapSizeBytes, aliasable);
  if (heap_ && heap_->totalSize() > 0) {
    useHeap_ = true;
    // Pin the heap arena into the residency set ONCE at heap creation
    // (long-lived registration).
    // Heap-vended id<MTLBuffer>s inherit residency from the parent
    // heap, so this single pin covers them all. The recorder still
    // records heap-vended buffers in per-CB binds_ and pinBatch them;
    // those calls become benign refcount bookkeeping (the heap-vended
    // buffer is already reachable via the heap's residency, so
    // addAllocation: is either a no-op or a duplicate-tracked entry;
    // either way the GPU sees the heap as resident).
    if (residency_ && heap_->nativeHeap()) {
      residency_->pinHeap(heap_->nativeHeap());
    }
    ET_LOG(Info, "MetalAllocator: heap enabled (%zu MB)", heapSizeBytes / (1024 * 1024));
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
