/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// MetalMTL3Backend implementation.
// Legacy Metal 3 dispatch backend. Owns the legacy MTLCommandBuffer +
// MTLComputeCommandEncoder + in-flight CB. Implements:
//   - ensureCommandBuffer / ensureEncoder / endEncoder
//   - setKernel / setBuffer / setBytes / dispatch (typed setters)
//   - commit / wait (submission + drain)
//   - ensureLegacyCommandBuffer / adoptLegacyCommandBuffer /
//     releaseLegacyCommandBuffer (MPS interop)
// Resource ownership convention (manual MRC):
//   - [queue_ commandBuffer] / [cb computeCommandEncoder] return AUTORELEASED
//     objects (Apple naming convention: not prefixed with "new"/"alloc"/"copy").
//   - We need to extend their lifetime past the autorelease pool drain, so we
//     [retain] explicitly. To prevent the +1 from autorelease leaking when no
//     enclosing pool drains in time, the call site is wrapped in
//     @autoreleasepool so the original +1 dies and only our retained +1 is
//     held. Cleanup is one [release].
//   - Audit Finding H7 / B6: previously the [retain] was paired without an
//     enclosing @autoreleasepool, leaving the autoreleased +1 to accumulate
//     across long-running processes. Now fixed.
//===----------------------------------------------------------------------===//

#import "MetalMTL3Backend.h"
#import "MetalStream.h"  // for MetalKernel::pipeline()

#include <executorch/runtime/platform/log.h>

#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

MetalMTL3Backend::MetalMTL3Backend(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue)
    : device_(device), queue_(queue) {
  // Borrowed refs — caller owns. No retain.
}

MetalMTL3Backend::~MetalMTL3Backend() {
  // Drain any in-flight work to avoid leaking the GPU's reference.
  if (inFlightCommandBuffer_) {
    [inFlightCommandBuffer_ waitUntilCompleted];
    [inFlightCommandBuffer_ release];
    inFlightCommandBuffer_ = nil;
  }
  if (encoder_) {
    [encoder_ endEncoding];
    [encoder_ release];
    encoder_ = nil;
  }
  if (commandBuffer_) {
    [commandBuffer_ release];
    commandBuffer_ = nil;
  }
}

//===----------------------------------------------------------------------===//
// Encoder lifecycle
//===----------------------------------------------------------------------===//

void MetalMTL3Backend::ensureCommandBuffer() {
  if (!commandBuffer_) {
    @autoreleasepool {
      // [queue_ commandBuffer] returns autoreleased; [retain] gives us +1
      // to extend the lifetime, and the @autoreleasepool drains the
      // autoreleased +1 at scope exit so it doesn't leak.
      commandBuffer_ = [[queue_ commandBuffer] retain];
    }
  }
}
namespace {
// Cached env-var probe: if ET_METAL_USE_METAL3_CONCURRENT=1, create the MTL3
// compute encoder in MTLDispatchTypeConcurrent mode. Default off — preserves
// today's serial-encoder semantics where Metal auto-orders dispatches.
// When concurrent mode is on:
//   - dispatches in the same encoder run in parallel UNLESS we explicitly
//     emit a [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]
//   - HazardTracker tells us when (via dispatchHazardAware's bool); the
//     conservative dispatch() emits one after EVERY dispatch (which
//     reproduces serial semantics with extra cost — only used by the
//     non-hazard-aware code paths).
// Pre-conditions for safety, all true today after recent refactors:
//   - All buffer binds go through typed setters (setInput/setOutput/setInOut/
//     setInputBuffer/setOutputBuffer) so HazardTracker sees every range.
//   - The bare hazard-blind setBuffer was removed from MetalStream.
//   - Subregion bindings resolve to (parent_mtl, offset) before the tracker
//     sees them, so split-K-style aliased writes are correctly detected.
bool useConcurrentEncoder() {
  static const bool enabled = []() {
    const char* env = std::getenv("ET_METAL_USE_METAL3_CONCURRENT");
    return env && env[0] == '1' && env[1] == '\0';
  }();
  return enabled;
}
}  // namespace

void MetalMTL3Backend::ensureEncoder() {
  if (!encoder_) {
    ensureCommandBuffer();
    @autoreleasepool {
      // computeCommandEncoder is autoreleased; retain + drain the
      // autorelease in this scope. When the env var is set we use the
      // explicit-dispatch-type form (Concurrent) so kernels can overlap
      // and the HazardTracker drives explicit barriers via
      // dispatchHazardAware below.
      if (useConcurrentEncoder()) {
        encoder_ = [[commandBuffer_
            computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
            retain];
      } else {
        encoder_ = [[commandBuffer_ computeCommandEncoder] retain];
      }
    }
  }
}

void MetalMTL3Backend::endEncoder() {
  if (encoder_) {
    [encoder_ endEncoding];
    [encoder_ release];
    encoder_ = nil;
  }
}

//===----------------------------------------------------------------------===//
// Per-dispatch encoding
//===----------------------------------------------------------------------===//

void MetalMTL3Backend::setKernel(MetalKernel* kernel) {
  ensureEncoder();
  [encoder_ setComputePipelineState:kernel->pipeline()];
}

void MetalMTL3Backend::setBuffer(
    id<MTLBuffer> buf,
    size_t offset,
    uint32_t slot) {
  ensureEncoder();
  [encoder_ setBuffer:buf offset:offset atIndex:slot];
}

void MetalMTL3Backend::setBytes(
    const void* ptr,
    size_t size,
    uint32_t slot) {
  ensureEncoder();
  [encoder_ setBytes:ptr length:size atIndex:slot];
}

void MetalMTL3Backend::dispatch(uvec3 grid, uvec3 block) {
  ensureEncoder();
  MTLSize mtlGrid = MTLSizeMake(grid.x, grid.y, grid.z);
  MTLSize mtlBlock = MTLSizeMake(block.x, block.y, block.z);
  [encoder_ dispatchThreadgroups:mtlGrid threadsPerThreadgroup:mtlBlock];
  if (useConcurrentEncoder()) {
    // Concurrent encoder: insert a buffer-scope barrier so the next
    // dispatch in this encoder sees this dispatch's writes. Conservative
    // — used by callers that don't analyze RAW/WAW hazards. Hazard-
    // analyzing callers should use dispatchHazardAware() to skip when
    // independence is proven.
    [encoder_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  // Serial encoder (default): Metal auto-orders dispatches; no barrier
  // needed.
}

void MetalMTL3Backend::dispatchHazardAware(
    uvec3 grid, uvec3 block, bool insertBarrierBefore) {
  ensureEncoder();
  if (useConcurrentEncoder() && insertBarrierBefore) {
    // Concurrent encoder + tracker says this dispatch reads/writes a
    // buffer that a prior pending dispatch wrote. Sync before encoding.
    [encoder_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  // Serial encoder: ignore the hint — auto-serialization handles it.
  MTLSize mtlGrid = MTLSizeMake(grid.x, grid.y, grid.z);
  MTLSize mtlBlock = MTLSizeMake(block.x, block.y, block.z);
  [encoder_ dispatchThreadgroups:mtlGrid threadsPerThreadgroup:mtlBlock];
}

//===----------------------------------------------------------------------===//
// Submission
//===----------------------------------------------------------------------===//

void MetalMTL3Backend::commit() {
  endEncoder();
  if (!commandBuffer_) {
    return;
  }
  if (inFlightCommandBuffer_) {
    // Caller flushed twice without an intervening wait(). Drain the older
    // submission first so we don't leak completion ownership of two cbufs.
    [inFlightCommandBuffer_ waitUntilCompleted];
    if ([inFlightCommandBuffer_ status] == MTLCommandBufferStatusError) {
      ET_LOG(
          Error,
          "MetalMTL3Backend: prior in-flight command buffer error: %s",
          [[inFlightCommandBuffer_ error] localizedDescription].UTF8String);
    }
    [inFlightCommandBuffer_ release];
    inFlightCommandBuffer_ = nil;
  }
  [commandBuffer_ commit];
  inFlightCommandBuffer_ = commandBuffer_;
  commandBuffer_ = nil;
}

void MetalMTL3Backend::wait() {
  if (!inFlightCommandBuffer_) {
    return;
  }
  [inFlightCommandBuffer_ waitUntilCompleted];
  if ([inFlightCommandBuffer_ status] == MTLCommandBufferStatusError) {
    ET_LOG(
        Error,
        "MetalMTL3Backend: command buffer error: %s",
        [[inFlightCommandBuffer_ error] localizedDescription].UTF8String);
  }
  [inFlightCommandBuffer_ release];
  inFlightCommandBuffer_ = nil;
}

//===----------------------------------------------------------------------===//
// MPS interop (legacy command buffer adoption)
//===----------------------------------------------------------------------===//

id<MTLCommandBuffer> MetalMTL3Backend::ensureLegacyCommandBuffer() {
  ensureCommandBuffer();
  return commandBuffer_;
}

void MetalMTL3Backend::adoptLegacyCommandBuffer(id<MTLCommandBuffer> newCB) {
  if (newCB == commandBuffer_) {
    return;
  }
  if (commandBuffer_) {
    [commandBuffer_ release];
    commandBuffer_ = nil;
  }
  if (newCB) {
    commandBuffer_ = newCB;
    [commandBuffer_ retain];
  }
}

void MetalMTL3Backend::releaseLegacyCommandBuffer() {
  if (commandBuffer_) {
    [commandBuffer_ release];
    commandBuffer_ = nil;
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
