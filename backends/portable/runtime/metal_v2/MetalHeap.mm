/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalStream.h"
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MetalHeap
//===----------------------------------------------------------------------===//

MetalHeap::MetalHeap(id<MTLDevice> device, size_t size, bool aliasable) : totalSize_(size) {
  MTLHeapDescriptor* desc = [[MTLHeapDescriptor alloc] init];
  desc.size = size;
  desc.storageMode = MTLStorageModeShared;  // Unified memory
  desc.cpuCacheMode = MTLCPUCacheModeDefaultCache;

  // Placement heap allows precise control over buffer placement
  desc.type = MTLHeapTypePlacement;

  // Hazard tracking mode
  if (@available(macOS 10.15, iOS 13.0, *)) {
    desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
  }

  // Aliasable resources can share memory (reduces footprint)
  if (@available(macOS 13.0, iOS 16.0, *)) {
    if (aliasable) {
      desc.type = MTLHeapTypeAutomatic;  // Allows aliasing
    }
  }

  heap_ = [device newHeapWithDescriptor:desc];
  [desc release];

  if (heap_) {
    [heap_ retain];
    ET_LOG(Info, "MetalHeap: Created %zu MB heap (aliasable=%d)", size / (1024*1024), aliasable);
  } else {
    ET_LOG(Error, "MetalHeap: Failed to create heap");
  }
}

MetalHeap::~MetalHeap() {
  if (heap_) {
    [heap_ release];
  }
}

id<MTLBuffer> MetalHeap::allocBuffer(size_t size) {
  if (!heap_) return nil;

  // Check if heap has space
  if (usedSize_ + size > totalSize_) {
    ET_LOG(Info, "MetalHeap: Out of space (need %zu, have %zu)",
           size, totalSize_ - usedSize_);
    return nil;
  }

  id<MTLBuffer> buffer = [heap_ newBufferWithLength:size
                                            options:MTLResourceStorageModeShared];
  if (buffer) {
    usedSize_ += [buffer allocatedSize];
    ET_LOG(Debug, "MetalHeap: Allocated %zu bytes (used: %zu/%zu)",
           size, usedSize_, totalSize_);
  }

  return buffer;
}


} // namespace metal_v2
} // namespace backends
} // namespace executorch

