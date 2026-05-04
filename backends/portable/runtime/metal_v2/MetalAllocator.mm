/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalAllocator.h"
#import "HazardTracker.h"

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

MetalAllocator::MetalAllocator(id<MTLDevice> device, HazardTracker* hazards)
    : device_(device), hazards_(hazards) {
  pool_ = std::make_unique<MetalBufferPool>(device_);
  // Metal 4: ResidencySet for GPU-resident memory. The manager handles
  // OS-availability gating internally; if the API isn't available it's
  // a no-op.
  residency_ = std::make_unique<ResidencyManager>(device_);
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

  if (residency_) residency_->add(buffer);

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
    if (residency_) residency_->remove(removed->mtl);
  }
  [removed->mtl release];
}

//===----------------------------------------------------------------------===//
// External buffer registration
//===----------------------------------------------------------------------===//

bool MetalAllocator::registerExternalBuffer(
    void* ptr, size_t bytes, bool strict_zero_copy) {
  if (!ptr || bytes == 0) return false;

  // Cache hit: refresh the snapshot for ExternalCopied entries.
  if (auto* entry = buffers_.find(ptr)) {
    (void)entry;
    buffers_.refreshIfCopied(ptr, bytes);
    return true;
  }

  // Page alignment check (16 KB on ARM64).
  bool pageAligned = ((uintptr_t)ptr % 16384) == 0;
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

  buffers_.insert(ptr, buffer, origin, bytes);
  if (residency_) residency_->add(buffer);

  ET_LOG(Debug, "MetalAllocator: registered %p -> MTLBuffer %p (origin=%d)",
         ptr, (__bridge void*)buffer, (int)origin);
  return true;
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
  }
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
    ET_LOG(Info, "MetalAllocator: heap enabled (%zu MB)", heapSizeBytes / (1024 * 1024));
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
