/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// BufferRegistry — single source of truth for MTLBuffer ownership.
// Replaces the three hand-maintained sets that previously lived on
// MetalStream:
//   - ptrToBuffer_       : void* -> MTLBuffer    (the actual map)
//   - externalBuffers_   : set<void*>            (external vs allocated)
//   - copiedBuffers_     : set<void*>            (snapshot vs zero-copy alias)
// Maintaining 3 invariants by hand at 4 mutation sites was the source of
// audit Finding C1 (free() check-after-erase: external buffers wrongly
// entered the pool, causing silent corruption). After this refactor, the
// (origin, mtl, size) tuple lives on a single Entry and routing decisions
// happen via a single map lookup — invariant drift becomes structurally
// impossible.
// What lives here:
//   - The (ptr -> Entry) map.
//   - An Origin tag on each entry distinguishing how the buffer was
//     created — used to route free() (Pool entries return to pool;
//     External entries don't).
//   - Refresh-on-hit logic for ExternalCopied entries (so stack-local
//     std::vector data passed via In{} stays correct on subsequent calls).
// What does NOT live here:
//   - The buffer pool itself (still owned by MetalStream as MetalBufferPool).
//     The registry is told "this came from the pool" via Origin::Pool, but
//     it doesn't know how to acquire/release pool buffers — the caller does.
//   - Residency-set membership (still an MTL4-side concern on MetalStream).
//   - The actual newBufferWithBytesNoCopy / newBufferWithBytes calls
//     (caller decides which path; passes the resulting MTLBuffer + Origin
//     to insert()).
// Thread safety: NOT thread-safe. MetalStream is documented as single-
// threaded for dispatches; the registry inherits that contract.
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

class BufferRegistry {
 public:
  // How the MTLBuffer was created. Routes free() and informs lookup behavior.
  enum class Origin {
    // Allocated via MetalBufferPool. free() should return to the pool.
    Pool,
    // allocated via MetalHeap (bump-allocator carved out of an
    // MTLHeap). free() does NOT return memory to the heap (the heap is
    // a single arena reset wholesale by MetalHeap::reset()) — it only
    // drops the +1 retain on the MTLBuffer wrapper. Tagging this
    // separately from Pool prevents heap-allocated buffers from being
    // mistakenly fed into MetalBufferPool::release(), which would
    // corrupt the pool's size buckets / cache them as if they came from
    // newBufferWithLength.
    Heap,
    // External caller memory wrapped via newBufferWithBytesNoCopy.
    // The MTLBuffer aliases the caller's pointer (live link). free() just
    // drops the wrapper; caller still owns the underlying memory.
    ExternalAliased,
    // External caller memory snapshotted via newBufferWithBytes (fallback
    // when newBufferWithBytesNoCopy fails — typically for non-page-aligned
    // pointers). The MTLBuffer holds a COPY made at registration time.
    // refreshIfCopied() must be called on subsequent uses to re-sync.
    ExternalCopied,
    // R8.3: Sub-region view into a parent MTLBuffer. The Entry borrows the
    // parent's MTLBuffer (no extra retain) and stores byte offset within
    // it. Used by aoti_torch__reinterpret_tensor and runtime_v2 MetalBuffer
    // subregion factories. Hazard tracking keys on the parent's MTLBuffer
    // identity + (offset, offset+size) range — lets two views into the
    // same workspace correctly detect overlap.
    Subregion,
  };

  struct Entry {
    id<MTLBuffer> mtl;  // retained by the registry (Pool/External*); BORROWED for Subregion
    Origin origin;
    size_t size;
    // R8.3: byte offset within `mtl`. Always 0 for Pool/External* (the entry
    // owns the whole MTLBuffer); set to the view offset for Subregion.
    size_t offset = 0;
  };

  BufferRegistry() = default;
  ~BufferRegistry();

  BufferRegistry(const BufferRegistry&) = delete;
  BufferRegistry& operator=(const BufferRegistry&) = delete;

  // Insert a new entry. The registry takes an additional retain on `mtl`
  // (caller is expected to have at least one retain when calling).
  // Inserting the same ptr twice is undefined — caller should remove() first.
  void insert(void* ptr, id<MTLBuffer> mtl, Origin origin, size_t size);

  // R8.3: insert a Subregion entry. Borrows the parent's MTLBuffer (does
  // NOT take an additional retain — the parent's entry owns it). On
  // remove() of a Subregion entry, no release is performed. Caller must
  // ensure the parent entry outlives all Subregion entries pointing to it.
  void insertSubregion(
      void* child_ptr,
      id<MTLBuffer> parent_mtl,
      size_t offset,
      size_t size);

  // Lookup. Returns nullptr if not registered.
  const Entry* find(void* ptr) const;

  // Convenience: returns nil if not registered.
  id<MTLBuffer> findBuffer(void* ptr) const;

  // Returns true if the registry has an entry for ptr.
  bool contains(void* ptr) const { return map_.count(ptr) > 0; }

  // Remove and return the entry. Caller is responsible for releasing the
  // MTLBuffer (and routing it to the pool, if Origin::Pool). Returns
  // nullopt if not registered.
  // After remove(), the registry has dropped its retain — caller's MTLBuffer
  // pointer in the returned Entry holds the remaining +1 ownership.
  std::optional<Entry> remove(void* ptr);

  // For ExternalCopied entries, copy `size` bytes from ptr into the
  // MTLBuffer's contents. No-op for other Origins or if ptr isn't
  // registered. Hard-fails (ET_CHECK) if size > entry.size — the
  // caller is asking for a refresh larger than the snapshot can hold,
  // and silently truncating would let the kernel read stale / garbage
  // data past entry.size. (B4 audit fix.)
  // Used to keep snapshot entries in sync when the caller's underlying
  // memory has changed since first registration (e.g. stack-local
  // std::vector reused across dispatches with different shapes).
  void refreshIfCopied(void* ptr, size_t size);

  // Iterate all entries. Used for residency-set updates and debug dumps.
  // Lambda must not insert/remove entries during iteration.
  void forEach(const std::function<void(void* ptr, const Entry&)>& fn) const;

  // Drop all entries. Releases each MTLBuffer (the registry's retain).
  // Caller is responsible for any pool-routing it wanted to do (this is
  // typically called from MetalStream::~MetalStream after pool teardown).
  void clear();

  size_t size() const { return map_.size(); }

 private:
  std::unordered_map<void*, Entry> map_;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
