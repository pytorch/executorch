/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/metal/core/HazardTracker.h>
#include <executorch/backends/metal/core/MetalAllocator.h>
#include <executorch/backends/metal/core/MetalCommandRecorder.h>
#include <executorch/backends/metal/core/MetalKernel.h>
#include <executorch/backends/metal/core/MetalConfig.h>  // ET_METAL4_AVAILABLE / ET_METAL4_ENABLE
#include <executorch/backends/metal/core/MetalKernelCompiler.h>
#include <executorch/backends/metal/core/MetalTypes.h>
#include <executorch/backends/metal/core/MpsInterop.h>  // for ET_METAL_USE_MPSGRAPH

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdlib>  // for getenv
#include <cstring>  // for strcmp
#include <memory>

namespace executorch {
namespace backends {
namespace metal_v2 {

#if ET_METAL_USE_MPSGRAPH
class MpsInterop;
#endif

/// True iff the Metal 4 dispatch paths are both compiled in AND the runtime
/// OS supports them AND the runtime opt-in is enabled. Gated by
/// ET_METAL4_ENABLE + macOS 26+ + the METAL_USE_MTL4 env var
/// (default ON when both prior gates pass). Cached on first call.
/// Tests can call resetUseMTL4ForTesting() to force a re-probe.
bool useMTL4();
void resetUseMTL4ForTesting();

/// Pure mapping from Apple-silicon arch-suffix character to the default
/// auto-flush interval (dispatches per command buffer). Exposed so unit
/// tests can probe each known suffix (p/g/s/d) without faking a device
/// name. Falls back to 40 for unknown suffixes.
int defaultFlushIntervalForArchSuffix(char suffix);

//===----------------------------------------------------------------------===//
// MetalStream — composes the per-thread Metal subsystems.
// Owns (in this construction order):
//   1. device_, queue_                 (immutable Metal handles)
//   2. compiler_   : MetalKernelCompiler
//   3. hazards_    : HazardTracker      (borrowed by allocator_ and recorder_)
//   4. allocator_  : MetalAllocator     (borrows hazards_)
//   5. recorder_   : MetalCommandRecorder (borrows allocator_, compiler_, hazards_)
//   6. mpsInterop_ : MpsInterop
//
// Canonical use:
//   stream->allocator().alloc(N)                       // memory
//   stream->recorder().beginDispatch(kernel)           // dispatch (RAII)
//       .setInput(...).setOutput(...).run(grid, block);
//   stream->compiler()->compile(...)                   // shader compile
//   stream->mps().encodeWithLegacyCommandBuffer(...)   // MPSGraph (if enabled)
//
// Stream-level convenience methods:
//   sync() / flush() / wait() — orchestrate across recorder + MpsInterop.
//
// Threading model:
//   MetalStream::get() returns a thread-local stream — each thread that
//   calls it gets its OWN MetalStream (and thus its own MetalAllocator,
//   BufferRegistry, ResidencyManager, MetalCommandRecorder, MpsInterop).
//   MetalKernelCache is the only piece of state shared across threads
//   (process-wide singleton in MetalKernelCache.cpp).
//
//   Consequences:
//   - The same external pointer registered from two threads creates two
//     independent registry entries and two residency-set memberships
//     (one per stream). Correctness is fine for ExternalAliased (the
//     OS-level MTLBuffer wrapping the same memory is bound twice), but
//     wasteful in registry/residency-set capacity.
//   - Cross-thread sharing of buffers should go through explicit
//     stream creation (MetalStream::create()) and disciplined ownership;
//     don't try to bind the same alloc()'d ptr from two threads.
//   - For multi-thread perf, a future iteration could hoist
//     BufferRegistry + ResidencyManager to a process-wide store (with
//     locking). Today's tradeoff favors lock-free per-thread state.
//===----------------------------------------------------------------------===//

class MetalStream {
 public:
  MetalStream();
  ~MetalStream();

  MetalStream(const MetalStream&) = delete;
  MetalStream& operator=(const MetalStream&) = delete;

  //===--------------------------------------------------------------------===//
  // Static factories
  //===--------------------------------------------------------------------===//

  /// Get the thread-local MetalStream — the safe default.
  static MetalStream* get();

  /// Create a new independent stream (caller owns lifetime).
  static std::unique_ptr<MetalStream> create();

  //===--------------------------------------------------------------------===//
  // Subsystem accessors
  //===--------------------------------------------------------------------===//

  MetalAllocator&        allocator() { return *allocator_; }
  const MetalAllocator&  allocator() const { return *allocator_; }
  MetalCommandRecorder&  recorder()  { return *recorder_; }
  const MetalCommandRecorder& recorder() const { return *recorder_; }
  MetalKernelCompiler*   compiler()  { return compiler_.get(); }
#if ET_METAL_USE_MPSGRAPH
  MpsInterop&            mps()       { return *mpsInterop_; }
  const MpsInterop&      mps() const { return *mpsInterop_; }
#endif

  id<MTLDevice>          device() const { return device_; }

  //===--------------------------------------------------------------------===//
  // Cross-subsystem orchestration
  //   sync() = flush + wait. flush() drains the recorder; wait() blocks
  //   until all submitted work (recorder CBs + any MPS-side commits) has
  //   GPU-completed.
  //===--------------------------------------------------------------------===//

  void flush() { recorder_->flush(); }
  void wait() { recorder_->wait(); }
  void sync() { flush(); wait(); }

 private:
  // Device flush interval based on architecture. Used by ctor to seed the
  // Recorder's flushInterval_ before any dispatch lands.
  int getDefaultFlushInterval() const;

  // Owned subsystems (in construction order):
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  std::unique_ptr<MetalKernelCompiler>  compiler_;
  std::unique_ptr<HazardTracker>        hazards_;
  std::unique_ptr<MetalAllocator>       allocator_;
  std::unique_ptr<MetalCommandRecorder> recorder_;
#if ET_METAL_USE_MPSGRAPH
  std::unique_ptr<MpsInterop>           mpsInterop_;
#endif
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
