/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ResidencyManager.h"

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

ResidencyManager::ResidencyManager(id<MTLDevice> device) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
    desc.label = @"GpuStream ResidencySet";
    desc.initialCapacity = 64;

    NSError* error = nil;
    id<MTLResidencySet> set =
        [device newResidencySetWithDescriptor:desc error:&error];
    [desc release];

    if (set) {
      set_ = set;  // takes the +1 retain from -newResidencySetWithDescriptor:
      enabled_ = true;
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
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (set_) [set_ release];
  }
#endif
  set_ = nil;
}

void ResidencyManager::add(id<MTLBuffer> buffer) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (enabled_ && set_ && buffer) {
      id<MTLResidencySet> rs = (id<MTLResidencySet>)set_;
      [rs addAllocation:buffer];
    }
  }
#else
  (void)buffer;
#endif
}

void ResidencyManager::remove(id<MTLBuffer> buffer) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (enabled_ && set_ && buffer) {
      id<MTLResidencySet> rs = (id<MTLResidencySet>)set_;
      [rs removeAllocation:buffer];
    }
  }
#else
  (void)buffer;
#endif
}

void ResidencyManager::commit() {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (enabled_ && set_) {
      id<MTLResidencySet> rs = (id<MTLResidencySet>)set_;
      [rs commit];
      [rs requestResidency];
      ET_LOG(Debug, "ResidencyManager: Committed (size=%llu bytes)",
             (unsigned long long)[rs allocatedSize]);
    }
  }
#endif
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
