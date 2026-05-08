/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MetalStream — composes the per-thread Metal subsystems:
//   - MetalAllocator         (memory)
//   - MetalCommandRecorder   (encoder + dispatch + hazards)
//   - MpsInterop             (MPS interop)
//   - MetalKernelCompiler    (compile)
// This file holds the methods that need a non-inline body:
//   - factories: get(), create()
//   - lifecycle: ctor, dtor
//   - getDefaultFlushInterval — arch-suffix dispatch heuristic
//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt).

#import "MetalStream.h"
#import "MetalMTL3Backend.h"
#import "MetalMTL4Backend.h"  // even with MTL4 OFF, class declaration exists
#import "MpsInterop.h"
#include <executorch/runtime/platform/log.h>

#include <mutex>

namespace executorch {
namespace backends {
namespace metal_v2 {

namespace {
std::mutex s_useMTL4_mu;
bool s_useMTL4_initialized = false;
bool s_useMTL4_value = false;
}  // namespace

bool useMTL4() {
#if ET_METAL4_ENABLE
  if (@available(macOS 26.0, iOS 26.0, *)) {
    std::lock_guard<std::mutex> lk(s_useMTL4_mu);
    if (!s_useMTL4_initialized) {
      // Default: enabled when both compile-time and OS-availability
      // gates pass. Env var explicitly turns it off (or on, for symmetry).
      const char* env = getenv("METAL_USE_MTL4");
      s_useMTL4_value = true;
      if (env) {
        if (strcmp(env, "0") == 0 || strcmp(env, "false") == 0 ||
            strcmp(env, "FALSE") == 0 || strcmp(env, "off") == 0 ||
            strcmp(env, "OFF") == 0) {
          s_useMTL4_value = false;
        } else if (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 ||
                   strcmp(env, "TRUE") == 0 || strcmp(env, "on") == 0 ||
                   strcmp(env, "ON") == 0) {
          s_useMTL4_value = true;
        }
        // Any other value: leave at default (true).
      }
      s_useMTL4_initialized = true;
    }
    return s_useMTL4_value;
  }
#endif
  return false;
}

void resetUseMTL4ForTesting() {
  std::lock_guard<std::mutex> lk(s_useMTL4_mu);
  s_useMTL4_initialized = false;
  s_useMTL4_value = false;
}

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
    // Hard-fail on missing Metal device — there's no usable recovery
    // path (no GPU, no fallback). Better to crash with a clear message
    // here than ship a half-built stream that crashes later in some op.
    device_ = MTLCreateSystemDefaultDevice();
    ET_CHECK_MSG(device_ != nil,
                 "MetalStream: MTLCreateSystemDefaultDevice() returned nil "
                 "(no Metal device available?)");

    queue_ = [device_ newCommandQueue];
    ET_CHECK_MSG(queue_ != nil,
                 "MetalStream: [device newCommandQueue] returned nil");

    // Subsystems in dependency order:
    //   compiler   → no deps
    //   hazards    → no deps; borrowed by allocator + recorder
    //   allocator  → device + hazards
    //   recorder   → device + queue + allocator + compiler + hazards
    //                (internally owns MTL3 backend + lazy MTL4 backend)
    //   mpsInterop → recorder + queue
    compiler_  = std::make_unique<MetalKernelCompiler>(device_);
    hazards_   = std::make_unique<HazardTracker>();
    allocator_ = std::make_unique<MetalAllocator>(device_, hazards_.get());
    recorder_  = std::make_unique<MetalCommandRecorder>(
        device_, queue_, allocator_.get(), compiler_.get(), hazards_.get());
    // Seed the recorder's auto-flush threshold from the device-arch heuristic.
    recorder_->setFlushInterval(getDefaultFlushInterval());

#if ET_METAL_USE_MPSGRAPH
    mpsInterop_ = std::make_unique<MpsInterop>(recorder_.get());
#endif

    ET_LOG(Info, "MetalStream: initialized with device '%s', flush interval=%d",
           [[device_ name] UTF8String], recorder_->flushInterval());
  }
}

MetalStream::~MetalStream() {
  @autoreleasepool {
    sync();  // flush() + wait() — drains any pending and in-flight work.

    // Tear-down order is the reverse of construction:
    //   mpsInterop → recorder → allocator → hazards → compiler → queue/device
#if ET_METAL_USE_MPSGRAPH
    mpsInterop_.reset();
#endif
    recorder_.reset();
    allocator_.reset();
    hazards_.reset();
    compiler_.reset();

    // ARC releases __strong queue_ and device_ when this object is destroyed.

    ET_LOG(Debug, "MetalStream: Destroyed");
  }
}

//===----------------------------------------------------------------------===//
// Device-arch heuristic for default flush interval.
//===----------------------------------------------------------------------===//

// Pure mapping from arch-suffix character to default flush interval.
// Exposed so unit tests can probe each known suffix without faking a
// device name.
int defaultFlushIntervalForArchSuffix(char suffix) {
  switch (suffix) {
    case 'p': return 20;  // iPhone — more conservative
    case 'g': return 40;  // Base/Pro
    case 's': return 50;  // Max
    case 'd': return 50;  // Ultra
    default:  return 40;
  }
}

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

  return defaultFlushIntervalForArchSuffix(suffix);
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
