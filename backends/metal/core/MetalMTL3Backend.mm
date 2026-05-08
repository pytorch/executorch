/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// MetalMTL3Backend implementation.
// Metal 3 dispatch backend. Owns the MTLCommandBuffer +
// MTLComputeCommandEncoder + in-flight CB. Implements:
//   - ensureCommandBuffer / ensureEncoder / endEncoder
//   - setKernel / setBuffer / setBytes / dispatch (typed setters)
//   - commit / wait (submission + drain)
//   - ensureLegacyCommandBuffer / adoptLegacyCommandBuffer /
//     releaseLegacyCommandBuffer (MPS interop)
//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). All
// id<...> ivars (commandBuffer_, encoder_, inFlightCommandBuffer_) are
// private to this TU and __strong by default. ARC retains via assignment
// from the autoreleased -[MTLCommandQueue commandBuffer] /
// -[MTLCommandBuffer computeCommandEncoder] return values, and releases
// on dtor or on nil-reassignment.
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
  // Borrowed refs — caller (MetalStream) owns lifetime. Under ARC the
  // ivars are __strong by default, so the assignments above retain;
  // both refs release on dtor.
}

MetalMTL3Backend::~MetalMTL3Backend() {
  // Drain any in-flight work to avoid leaking the GPU's reference.
  if (inFlightCommandBuffer_) {
    [inFlightCommandBuffer_ waitUntilCompleted];
    inFlightCommandBuffer_ = nil;
  }
  if (encoder_) {
    [encoder_ endEncoding];
    encoder_ = nil;
  }
  // commandBuffer_ released by ARC's implicit dtor logic.
}

//===----------------------------------------------------------------------===//
// Encoder lifecycle
//===----------------------------------------------------------------------===//

void MetalMTL3Backend::ensureCommandBuffer() {
  if (!commandBuffer_) {
    commandBuffer_ = [queue_ commandBuffer];
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
// Pre-conditions for safety:
//   - All buffer binds go through typed setters (setInput/setOutput/setInOut/
//     setInputBuffer/setOutputBuffer) so HazardTracker sees every range.
//   - The Recorder exposes no hazard-blind setBuffer entry point.
//   - Subregion bindings resolve to (parent_mtl, offset) before the tracker
//     sees them, so split-K-style aliased writes are correctly detected.
// Concurrent-encoder probe state. File-static (not function-local) so
// resetUseConcurrentEncoderForTesting() below can clear it.
std::mutex s_useConcurrent_mu;
bool s_useConcurrent_initialized = false;
bool s_useConcurrent_enabled = false;

bool useConcurrentEncoder() {
  std::lock_guard<std::mutex> lk(s_useConcurrent_mu);
  if (!s_useConcurrent_initialized) {
    const char* env = std::getenv("ET_METAL_USE_METAL3_CONCURRENT");
    s_useConcurrent_enabled = env && env[0] == '1' && env[1] == '\0';
    s_useConcurrent_initialized = true;
  }
  return s_useConcurrent_enabled;
}
}  // namespace

void resetUseConcurrentEncoderForTesting() {
  std::lock_guard<std::mutex> lk(s_useConcurrent_mu);
  s_useConcurrent_initialized = false;
  s_useConcurrent_enabled = false;
}

void MetalMTL3Backend::ensureEncoder() {
  if (!encoder_) {
    ensureCommandBuffer();
    if (useConcurrentEncoder()) {
      encoder_ = [commandBuffer_
          computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    } else {
      encoder_ = [commandBuffer_ computeCommandEncoder];
    }
  }
}

void MetalMTL3Backend::endEncoder() {
  if (encoder_) {
    [encoder_ endEncoding];
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
  // Apple's MTLComputeCommandEncoder retains buffers it binds and
  // releases them at endEncoding (and per-rebind). Nothing for us
  // to do beyond the API call.
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
    // Reached only when commit() is called with no encoded work. The
    // per-CB unpin path in MetalCommandRecorder::flush guards its
    // pending-handler increment behind binds_ non-empty, which implies
    // a CB is live by construction — so this branch is unreachable from
    // any path that has registered a completion handler.
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
    inFlightCommandBuffer_ = nil;
  }
  // Per the per-CB unpin contract: attach any pending completion handlers to
  // this CB BEFORE commit so the handler runs when the CB completes
  // (driver thread). Move-capture into the Obj-C block so the
  // std::function lifetime is owned by the block (released when block
  // tears down after invocation).
  if (!pendingCompletionHandlers_.empty()) {
    auto handlers = std::move(pendingCompletionHandlers_);
    pendingCompletionHandlers_.clear();
    [commandBuffer_ addCompletedHandler:^(id<MTLCommandBuffer> /*cb*/) {
      for (auto& h : handlers) {
        if (h) h();
      }
    }];
  }
  [commandBuffer_ commit];
  // Move ownership: inFlightCommandBuffer_ takes the +1, commandBuffer_
  // is nil-reassigned to drop the original ref.
  inFlightCommandBuffer_ = commandBuffer_;
  commandBuffer_ = nil;
}

void MetalMTL3Backend::addCompletionHandler(std::function<void()> handler) {
  if (handler) {
    pendingCompletionHandlers_.push_back(std::move(handler));
  }
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
  commandBuffer_ = newCB;
}

void MetalMTL3Backend::releaseLegacyCommandBuffer() {
  commandBuffer_ = nil;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
