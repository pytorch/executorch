/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// MetalMTL4Backend implementation.
// All MTL4-specific state and logic. Compiles to no-op stubs when
// ET_METAL4_ENABLE=0 (the default). When enabled and the runtime OS
// supports Metal 4 (macOS 26.0+ / iOS 26.0+), MetalStream constructs an
// instance that owns the MTL4 queue, allocator, arg table, scratch
// buffer, completion event, and active CB/encoder.
//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). All
// id<...> ivars are private to this TU and __strong by default; ARC
// retains on assignment from the +1-returning new* methods and releases
// on dtor (or on reassignment to nil). Transient descriptors
// (MTL4CommandQueueDescriptor, MTL4CommandAllocatorDescriptor,
// MTL4ArgumentTableDescriptor, MTL4CompilerDescriptor, MTL4CommitOptions)
// are __strong locals; ARC releases them at scope exit.
//===----------------------------------------------------------------------===//

#import "MetalMTL4Backend.h"
#import "MetalStream.h"  // for MetalStream + MetalKernel access
#import "ResidencyManager.h"

#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstring>

namespace executorch {
namespace backends {
namespace metal_v2 {

#if ET_METAL4_ENABLE

//===----------------------------------------------------------------------===//
// Constructor / destructor
//===----------------------------------------------------------------------===//

MetalMTL4Backend::MetalMTL4Backend(id<MTLDevice> device,
                                   ResidencyManager* residency,
                                   size_t scalarScratchBytes)
    : device_(device), residency_(residency) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      NSError* err = nil;

      // Command queue.
      MTL4CommandQueueDescriptor* qDesc = [[MTL4CommandQueueDescriptor alloc] init];
      mtl4Queue_ = [device_ newMTL4CommandQueueWithDescriptor:qDesc error:&err];
      if (!mtl4Queue_ || err) {
        ET_LOG(Error, "MetalMTL4Backend: MTL4CommandQueue creation failed: %s",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        mtl4Queue_ = nil;
        return;
      }

      // Command allocator
      err = nil;
      MTL4CommandAllocatorDescriptor* aDesc = [[MTL4CommandAllocatorDescriptor alloc] init];
      mtl4Allocator_ = [device_ newCommandAllocatorWithDescriptor:aDesc error:&err];
      if (!mtl4Allocator_ || err) {
        ET_LOG(Error, "MetalMTL4Backend: MTL4CommandAllocator creation failed: %s",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        mtl4Queue_ = nil;
        return;
      }

      // Argument table (single, reused per dispatch)
      MTL4ArgumentTableDescriptor* atDesc = [[MTL4ArgumentTableDescriptor alloc] init];
      atDesc.maxBufferBindCount = MetalCommandRecorder::kMaxBuffersPerDispatch;
      err = nil;
      mtl4ArgTable_ = [device_ newArgumentTableWithDescriptor:atDesc error:&err];
      if (!mtl4ArgTable_ || err) {
        ET_LOG(Error, "MetalMTL4Backend: MTL4ArgumentTable creation failed: %s",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        mtl4Allocator_ = nil;
        mtl4Queue_ = nil;
        return;
      }

      // Inline-scalar bump scratch (caller-configurable, default 1 MB).
      mtl4ScalarScratch_ = [device_ newBufferWithLength:scalarScratchBytes
                                                options:MTLResourceStorageModeShared];
      // Register scratch with the residency set so MTL4 commands see
      // it as resident.
      if (residency_) residency_->add(mtl4ScalarScratch_);
      mtl4ScalarScratchOffset_ = 0;

      // Completion event for wait()
      mtl4CompletionEvent_ = [device_ newSharedEvent];
      mtl4CompletionValue_ = 0;

      ET_LOG(Info, "MetalMTL4Backend: initialized (queue+allocator+arg-table+scratch+event)");
    }
  }
}

MetalMTL4Backend::~MetalMTL4Backend() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4InFlightCommandBuffer_) {
      // Best-effort drain of in-flight work before teardown.
      if (mtl4CompletionValue_ > 0) {
        [mtl4CompletionEvent_ waitUntilSignaledValue:mtl4CompletionValue_
                                            timeoutMS:UINT64_MAX];
      }
      mtl4InFlightCommandBuffer_ = nil;
    }
    // Explicit nil-assignment keeps the teardown order visible.
    if (mtl4Encoder_) { mtl4Encoder_ = nil; }
    if (mtl4CommandBuffer_) { mtl4CommandBuffer_ = nil; }
    mtl4ArgTable_ = nil;
    mtl4Allocator_ = nil;
    // Detach the residency set from this backend's queue before
    // destroying the queue — symmetric pair to the addQueueResidency()
    // call wired from MetalCommandRecorder's ctor.
    if (residency_ && mtl4Queue_) {
      residency_->removeQueueResidency(mtl4Queue_);
    }
    mtl4Queue_ = nil;
    mtl4ScalarScratch_ = nil;
    mtl4CompletionEvent_ = nil;
    mtl4Compiler_ = nil;
  }
}

void MetalMTL4Backend::addQueueResidency(id<MTLResidencySet> set) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4Queue_ && set) {
      [mtl4Queue_ addResidencySet:set];
    }
  }
}

//===----------------------------------------------------------------------===//
// Encoder lifecycle
//===----------------------------------------------------------------------===//

void MetalMTL4Backend::ensureCommandBuffer() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4Queue_ && !mtl4CommandBuffer_) {
      mtl4CommandBuffer_ = [device_ newCommandBuffer];
      [mtl4CommandBuffer_ beginCommandBufferWithAllocator:mtl4Allocator_];
    }
  }
}

void MetalMTL4Backend::ensureEncoder() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4Queue_ && !mtl4Encoder_) {
      ensureCommandBuffer();
      mtl4Encoder_ = [mtl4CommandBuffer_ computeCommandEncoder];
      [mtl4Encoder_ setArgumentTable:mtl4ArgTable_];
    }
  }
}

void MetalMTL4Backend::endEncoder() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4Encoder_) {
      [mtl4Encoder_ endEncoding];
      mtl4Encoder_ = nil;
    }
  }
}

//===----------------------------------------------------------------------===//
// Per-dispatch encoding
//===----------------------------------------------------------------------===//

void MetalMTL4Backend::setKernel(MetalKernel* kernel) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    ensureEncoder();
    [mtl4Encoder_ setComputePipelineState:kernel->pipeline()];
  }
}

void MetalMTL4Backend::setBuffer(
    id<MTLBuffer> buf,
    size_t offset,
    uint32_t slot) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    ensureEncoder();
    MTLGPUAddress addr = [buf gpuAddress] + offset;
    [mtl4ArgTable_ setAddress:addr atIndex:slot];
  }
}

void MetalMTL4Backend::setBytes(
    const void* ptr,
    size_t size,
    uint32_t slot) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    ensureEncoder();
    constexpr size_t kAlign = 16;
    char* scratchPtr = (char*)[mtl4ScalarScratch_ contents];
    MTLGPUAddress scratchBase = [mtl4ScalarScratch_ gpuAddress];
    const size_t kScratchCap = (size_t)[mtl4ScalarScratch_ length];

    size_t off = (mtl4ScalarScratchOffset_ + kAlign - 1) & ~(kAlign - 1);
    ET_CHECK_MSG(
        off + size <= kScratchCap,
        "MetalMTL4Backend::setBytes: scratch exhausted (need %zu at offset %zu, "
        "cap %zu). Lower flushInterval or use a real MTLBuffer for this payload.",
        size, off, kScratchCap);
    std::memcpy(scratchPtr + off, ptr, size);
    mtl4ScalarScratchOffset_ = off + size;
    [mtl4ArgTable_ setAddress:(scratchBase + off) atIndex:slot];
  }
}

void MetalMTL4Backend::dispatch(uvec3 grid, uvec3 block) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    ensureEncoder();
    MTLSize mtlGrid = MTLSizeMake(grid.x, grid.y, grid.z);
    MTLSize mtlBlock = MTLSizeMake(block.x, block.y, block.z);
    // Re-bind the (now-mutated) argument table just before dispatch.
    // setArgumentTable: snapshots the table state — without re-binding
    // after our setAddress: mutations, the encoder would dispatch with
    // stale (encoder-creation-time) table contents.
    [mtl4Encoder_ setArgumentTable:mtl4ArgTable_];
    [mtl4Encoder_ dispatchThreadgroups:mtlGrid threadsPerThreadgroup:mtlBlock];
    // Insert a memory barrier so the *next* dispatch in this encoder
    // sees this dispatch's writes. MTL4 has no automatic hazard tracking.
    // Conservative — used by callers that don't analyze RAW/WAW hazards.
    // Hazard-analyzing callers should use dispatchHazardAware() to skip
    // the barrier when independence is proven.
    [mtl4Encoder_ barrierAfterEncoderStages:MTLStageDispatch
                        beforeEncoderStages:MTLStageDispatch
                          visibilityOptions:MTL4VisibilityOptionDevice];
  }
}

void MetalMTL4Backend::dispatchHazardAware(
    uvec3 grid, uvec3 block, bool insertBarrierBefore) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    ensureEncoder();
    if (insertBarrierBefore) {
      // Synchronize with prior dispatches in this encoder. Caller has
      // analyzed read/write sets and determined this dispatch reads/writes
      // a buffer that a prior pending dispatch wrote.
      [mtl4Encoder_ barrierAfterEncoderStages:MTLStageDispatch
                          beforeEncoderStages:MTLStageDispatch
                            visibilityOptions:MTL4VisibilityOptionDevice];
    }
    MTLSize mtlGrid = MTLSizeMake(grid.x, grid.y, grid.z);
    MTLSize mtlBlock = MTLSizeMake(block.x, block.y, block.z);
    [mtl4Encoder_ setArgumentTable:mtl4ArgTable_];
    [mtl4Encoder_ dispatchThreadgroups:mtlGrid threadsPerThreadgroup:mtlBlock];
    // NO post-barrier — caller is responsible for hazard analysis on the
    // NEXT dispatch (which will pre-barrier if needed). This is what
    // unlocks parallelism: independent consecutive dispatches run
    // concurrently on the GPU.
  }
}

//===----------------------------------------------------------------------===//
// Submission
//===----------------------------------------------------------------------===//

void MetalMTL4Backend::commit() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    endEncoder();
    if (!mtl4CommandBuffer_) {
      // See MetalMTL3Backend::commit() — same invariant: this branch is
      // unreachable from the per-CB unpin path because binds_ non-empty
      // implies a CB is live.
      return;
    }
    // Drain prior MTL4 in-flight buffer if caller flushed twice without wait().
    if (mtl4InFlightCommandBuffer_) {
      if (mtl4CompletionValue_ > 0) {
        [mtl4CompletionEvent_ waitUntilSignaledValue:mtl4CompletionValue_
                                             timeoutMS:UINT64_MAX];
      }
      mtl4InFlightCommandBuffer_ = nil;
    }
    // Order matters: signalEvent enqueues a signal that fires after all work
    // *previously* committed completes. Commit FIRST, then signalEvent.
    ++mtl4CompletionValue_;
    [mtl4CommandBuffer_ endCommandBuffer];
    const id<MTL4CommandBuffer> bufs[1] = { mtl4CommandBuffer_ };
    // Per the per-CB unpin contract: if any completion handlers were registered
    // for this CB, attach them via MTL4CommitOptions::addFeedbackHandler:.
    // Apple invokes the block on a Metal-internal queue after the CB
    // completes; ResidencyManager (the typical consumer of these handlers)
    // is process-wide and internally thread-safe, so calling unpinBatch
    // from there is well-defined.
    if (!pendingCompletionHandlers_.empty()) {
      auto handlers = std::move(pendingCompletionHandlers_);
      pendingCompletionHandlers_.clear();
      MTL4CommitOptions* opts = [[MTL4CommitOptions alloc] init];
      [opts addFeedbackHandler:^(id<MTL4CommitFeedback> /*fb*/) {
        for (auto& h : handlers) {
          if (h) h();
        }
      }];
      [mtl4Queue_ commit:bufs count:1 options:opts];
    } else {
      [mtl4Queue_ commit:bufs count:1];
    }
    [mtl4Queue_ signalEvent:mtl4CompletionEvent_ value:mtl4CompletionValue_];
    // Move ownership: mtl4InFlightCommandBuffer_ takes the +1,
    // mtl4CommandBuffer_ is nil-reassigned to drop the original ref.
    mtl4InFlightCommandBuffer_ = mtl4CommandBuffer_;
    mtl4CommandBuffer_ = nil;
    // Reset per-flush state for MTL4 path.
    mtl4ScalarScratchOffset_ = 0;
  }
}

void MetalMTL4Backend::addCompletionHandler(std::function<void()> handler) {
  if (handler) {
    pendingCompletionHandlers_.push_back(std::move(handler));
  }
}

void MetalMTL4Backend::wait() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4InFlightCommandBuffer_) {
      if (mtl4CompletionValue_ > 0) {
        [mtl4CompletionEvent_ waitUntilSignaledValue:mtl4CompletionValue_
                                            timeoutMS:UINT64_MAX];
      }
      mtl4InFlightCommandBuffer_ = nil;
    }
  }
}

//===----------------------------------------------------------------------===//

id<MTL4Compiler> MetalMTL4Backend::getOrCreateMTL4Compiler() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (!mtl4Compiler_) {
      MTL4CompilerDescriptor* compilerDesc = [[MTL4CompilerDescriptor alloc] init];
      NSError* compilerErr = nil;
      mtl4Compiler_ = [device_ newCompilerWithDescriptor:compilerDesc
                                                   error:&compilerErr];
      if (!mtl4Compiler_ || compilerErr) {
        ET_LOG(Error,
               "MetalMTL4Backend: MTL4Compiler creation failed: %s",
               compilerErr ? [[compilerErr localizedDescription] UTF8String]
                           : "unknown");
        mtl4Compiler_ = nil;
        return nil;
      }
    }
    return mtl4Compiler_;
  }
  return nil;
}

#endif  // ET_METAL4_ENABLE

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
