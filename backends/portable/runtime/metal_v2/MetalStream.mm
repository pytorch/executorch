/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MetalStream — slim facade composing the per-thread Metal subsystems.
// All dispatch / alloc / hazards work is now in the peer classes:
//   - MetalAllocator         (memory)
//   - MetalCommandRecorder   (encoder + dispatch + hazards)
//   - MpsInterop              (MPS interop)
//   - MetalKernelCompiler    (compile)
// Most methods on MetalStream are inline shims declared in MetalStream.h
// that delegate to the peer subsystems for back-compat. This file holds
// only the small set of methods that need a non-inline body:
//   - factories: get(), create()
//   - lifecycle: ctor, dtor
//   - wait()                — composes recorder.wait() + MPS-bridge drain
//   - encodeWithLegacyCommandBuffer — forwards to MpsInterop
//   - getDefaultFlushInterval — arch-suffix dispatch heuristic

#import "MetalStream.h"
#import "MetalMTL3Backend.h"
#import "MetalMTL4Backend.h"  // even with MTL4 OFF, class declaration exists
#import "MpsInterop.h"
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// Factories
//===----------------------------------------------------------------------===//

MetalStream* MetalStream::get() {
  // thread_local unique_ptr: constructed on first access per thread,
  // destroyed when the thread exits. No explicit synchronization needed
  // because every thread has its own storage.
  static thread_local std::unique_ptr<MetalStream> threadLocalStream =
      std::make_unique<MetalStream>();
  return threadLocalStream.get();
}

std::unique_ptr<MetalStream> MetalStream::create() {
  return std::make_unique<MetalStream>();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

MetalStream::MetalStream() {
  @autoreleasepool {
    // N1: hard-fail on missing Metal device — there's no usable recovery
    // path (no GPU, no fallback). Better to crash with a clear message
    // here than ship a half-built stream that crashes later in some op.
    device_ = MTLCreateSystemDefaultDevice();
    ET_CHECK_MSG(device_ != nil,
                 "MetalStream: MTLCreateSystemDefaultDevice() returned nil "
                 "(no Metal device available?)");
    [device_ retain];

    queue_ = [device_ newCommandQueue];
    ET_CHECK_MSG(queue_ != nil,
                 "MetalStream: [device newCommandQueue] returned nil");
    [queue_ retain];

    // Subsystems in dependency order:
    //   compiler  → no deps
    //   hazards   → no deps; borrowed by allocator + recorder
    //   allocator → device + hazards
    //   recorder  → device + queue + allocator + compiler + hazards
    //               (internally owns MTL3 backend + lazy MTL4 backend)
    //   mpsBridge → recorder's mtl3+mtl4 backends + queue + allocator residency
    compiler_  = std::make_unique<MetalKernelCompiler>(device_);
    hazards_   = std::make_unique<HazardTracker>();
    allocator_ = std::make_unique<MetalAllocator>(device_, hazards_.get());
    recorder_  = std::make_unique<MetalCommandRecorder>(
        device_, queue_, allocator_.get(), compiler_.get(), hazards_.get());
    // Seed the recorder's auto-flush threshold from the device-arch heuristic.
    recorder_->setFlushInterval(getDefaultFlushInterval());

    // MpsInterop ctor signature is unchanged from pre-Step-2; we plumb the
    // same data from the new owners. Step 3 will retire the stream*
    // pointer in favor of a recorder*.
#if ET_METAL_USE_MPSGRAPH
    mpsInterop_ = std::make_unique<MpsInterop>(recorder_.get(), queue_);
#endif

    ET_LOG(Info, "MetalStream: initialized with device '%s', flush interval=%d",
           [[device_ name] UTF8String], recorder_->flushInterval());
  }
}

MetalStream::~MetalStream() {
  @autoreleasepool {
    sync();  // flush() + wait() — drains any pending and in-flight work.

    // Tear-down order is the reverse of construction:
    //   mpsBridge → recorder → allocator → hazards → compiler → queue/device
#if ET_METAL_USE_MPSGRAPH
    mpsInterop_.reset();
#endif
    recorder_.reset();   // releases backend_ (MTL3) + mtl4Backend_ + psoWrapCache
    allocator_.reset();  // releases buffers + pool + heap + residency
    hazards_.reset();
    compiler_.reset();

    [queue_ release];
    [device_ release];

    ET_LOG(Debug, "MetalStream: Destroyed");
  }
}

//===----------------------------------------------------------------------===//
// MPSGraph integration. Just a thin forwarder to MpsInterop.
//===----------------------------------------------------------------------===//

#if ET_METAL_USE_MPSGRAPH
void MetalStream::encodeWithLegacyCommandBuffer(
    std::function<void(MPSCommandBuffer* mpsCB)> encode_fn) {
  mpsInterop_->encodeWithLegacyCommandBuffer(std::move(encode_fn));
}
#endif

//===----------------------------------------------------------------------===//
// Device-arch heuristic for default flush interval.
//===----------------------------------------------------------------------===//

int MetalStream::getDefaultFlushInterval() const {
  // Architecture string suffix: 'p' = iPhone, 'g' = base, 's' = Max, 'd' = Ultra
  char suffix = 'g';

  if (@available(macOS 13.0, iOS 16.0, *)) {
    id architecture = [device_ architecture];
    if (architecture) {
      NSString* name = [architecture name];
      if (name && [name length] > 0) {
        suffix = [name characterAtIndex:[name length] - 1];
      }
    }
  }

  switch (suffix) {
    case 'p': return 20;  // iPhone — more conservative
    case 'g': return 40;  // Base/Pro
    case 's': return 50;  // Max
    case 'd': return 50;  // Ultra
    default:  return 40;
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
