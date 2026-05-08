/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). The
// id<MTLHeap> heap_ ivar is private to this TU and __strong by default.
//
// allocBuffer() returns an autoreleased id<MTLBuffer>. Caller
// MetalAllocator::alloc retains via BufferRegistry::insert, so the
// autoreleased reference survives long enough for the registry to take
// its retain.
//===----------------------------------------------------------------------===//

#import "MetalStream.h"
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MetalHeap
//===----------------------------------------------------------------------===//

MetalHeap::MetalHeap(id<MTLDevice> device, size_t size, bool aliasable) : totalSize_(size) {
  ET_CHECK_MSG(
      !aliasable,
      "MetalHeap: aliasable=true is not currently supported. "
      "Wiring requires MTLHeapTypeAutomatic + MTLHazardTrackingModeUntracked + "
      "the no-offset newBufferWithLength:options: API. No caller exercises "
      "this path today; remove this guard and complete the wiring when needed.");

  MTLHeapDescriptor* desc = [[MTLHeapDescriptor alloc] init];
  desc.size = size;
  desc.storageMode = MTLStorageModeShared;  // Unified memory
  desc.cpuCacheMode = MTLCPUCacheModeDefaultCache;

  // Placement heap: caller controls each buffer's offset within the heap.
  desc.type = MTLHeapTypePlacement;

  // Hazard tracking mode
  if (@available(macOS 10.15, iOS 13.0, *)) {
    desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
  }

  heap_ = [device newHeapWithDescriptor:desc];

  if (heap_) {
    ET_LOG(Info, "MetalHeap: Created %zu MB heap", size / (1024*1024));
  } else {
    ET_LOG(Error, "MetalHeap: Failed to create heap");
  }
}

MetalHeap::~MetalHeap() = default;

id<MTLBuffer> MetalHeap::allocBuffer(size_t size) {
  if (!heap_) return nil;

  // Placement heaps require an explicit offset (the heap doesn't
  // suballocate for you — the caller decides where each buffer lives).
  // Use [device heapBufferSizeAndAlignWithLength:options:] to compute
  // the aligned size + alignment, round usedSize_ up to alignment, and
  // place the buffer at that offset.
  MTLResourceOptions opts = MTLResourceStorageModeShared;
  MTLSizeAndAlign req =
      [[heap_ device] heapBufferSizeAndAlignWithLength:size options:opts];
  // Round usedSize_ up to req.align.
  size_t alignment = req.align ? req.align : 1;
  size_t alignedOffset = (usedSize_ + alignment - 1) & ~(alignment - 1);

  // Check if heap has space.
  if (alignedOffset + req.size > totalSize_) {
    ET_LOG(Info, "MetalHeap: Out of space (need %zu+pad at offset %zu, total %zu)",
           req.size, alignedOffset, totalSize_);
    return nil;
  }

  id<MTLBuffer> buffer = [heap_ newBufferWithLength:size
                                            options:opts
                                             offset:alignedOffset];
  if (buffer) {
    usedSize_ = alignedOffset + [buffer allocatedSize];
    ET_LOG(Debug, "MetalHeap: Allocated %zu bytes at offset %zu (used: %zu/%zu)",
           size, alignedOffset, usedSize_, totalSize_);
  } else {
    ET_LOG(Error, "MetalHeap: newBufferWithLength:options:offset: returned nil "
                  "for size=%zu offset=%zu", size, alignedOffset);
  }

  return buffer;
}


} // namespace metal_v2
} // namespace backends
} // namespace executorch
