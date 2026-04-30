/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalMTL4Backend — Metal 4 implementation of IComputeBackend.
// Owns ALL MTL4-specific state that previously lived on MetalStream:
//   - mtl4Queue_                  : MTL4 command queue
//   - mtl4Allocator_              : MTL4 command allocator
//   - mtl4CommandBuffer_          : currently-recording MTL4 CB
//   - mtl4InFlightCommandBuffer_  : committed CB awaiting GPU completion
//   - mtl4Encoder_                : active MTL4 compute encoder
//   - mtl4ArgTable_               : reusable arg table (4KB scalars + buffers)
//   - mtl4ScalarScratch_          : 1MB bump-allocated buffer for inline data
//   - mtl4ScalarScratchOffset_    : write head into the scratch buffer
//   - mtl4CompletionEvent_        : MTLSharedEvent for wait()
//   - mtl4CompletionValue_        : monotonic counter signaled per commit
//   - mpsSyncEvent_               : cross-queue sync between MTL4 and MPS
// Borrows (does NOT retain):
//   - id<MTLDevice>               : owned by MetalStream
//   - back-reference to MetalStream for residency-set integration
//     (residency set is a cross-cutting concern shared with MTL3 backend
//     and external buffer registration; stays on MetalStream).
// Compiles only when EXECUTORCH_METAL4_ENABLE=ON (gated via ET_METAL4_ENABLE
// macro). When the flag is off, MetalMTL4Backend declarations are visible
// but the class isn't instantiated — MetalStream's mtl4Backend_ stays
// nullptr and all dispatches go through the MTL3 backend.
//===----------------------------------------------------------------------===//

#include <executorch/backends/portable/runtime/metal_v2/IComputeBackend.h>

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#if !defined(ET_METAL4_ENABLE)
#define ET_METAL4_ENABLE 0
#endif

namespace executorch {
namespace backends {
namespace metal_v2 {

// Forward declaration to avoid pulling MetalStream.h here. ResidencyManager
// is owned by MetalStream and a borrowed pointer is handed to us at ctor.
class MetalKernel;
class ResidencyManager;

#if ET_METAL4_ENABLE

class MetalMTL4Backend : public IComputeBackend {
 public:
  // device  : borrowed reference (caller owns lifetime).
  // stream  : borrowed reference for cross-cutting concerns
  //           (kernel cache lookup, scalar bytes scratch addressing).
  // residency : borrowed reference; if non-null, the scratch buffer we
  //             allocate is registered with it and the queue gets the
  //             set wired in via addQueueResidency(). Pass nullptr if
  //             the stream has no residency manager (older OS, etc.).
  MetalMTL4Backend(id<MTLDevice> device,
                   ResidencyManager* residency)
      API_AVAILABLE(macos(26.0), ios(26.0));
  ~MetalMTL4Backend() override API_AVAILABLE(macos(26.0), ios(26.0));

  MetalMTL4Backend(const MetalMTL4Backend&) = delete;
  MetalMTL4Backend& operator=(const MetalMTL4Backend&) = delete;

  // True iff setupMTL4 in the constructor succeeded fully.
  bool isReady() const { return mtl4Queue_ != nil; }

  // ---- IComputeBackend (per-dispatch encoding) ----
  void setKernel(MetalKernel* kernel) override;
  void setBuffer(id<MTLBuffer> buf, size_t offset, uint32_t slot) override;
  void setBytes(const void* ptr, size_t size, uint32_t slot) override;
  void dispatch(uvec3 grid, uvec3 block) override;
  void dispatchHazardAware(
      uvec3 grid, uvec3 block, bool insertBarrierBefore) override;

  // ---- IComputeBackend (encoder lifecycle) ----
  void endEncoder() override;

  // ---- IComputeBackend (submission) ----
  void commit() override;
  void wait() override;

  //===--------------------------------------------------------------------===//

  // Get-or-create the MTL4 shader compiler. Lazy because not all sessions
  // compile kernels immediately, and the compiler is a heavyweight object
  // (~tens of ms first-time setup). Reused across all PSO creations.
  // Returns nil if MTL4 isn't available at runtime.
  // previously MetalKernelCompiler owned this field
  // directly. Moving it to MetalMTL4Backend (the right architectural
  // owner of MTL4-specific resources) means MetalKernelCompiler can be
  // pure MTL3-or-MTL4-agnostic and only reach into the MTL4 backend
  // when actually compiling under MTL4.
  id<MTL4Compiler> getOrCreateMTL4Compiler();

  // Internal helpers exposed for MetalStream to drive the MTL4 dispatch
  // path during ensureCommandBuffer / ensureEncoder calls.
  void ensureCommandBuffer();
  void ensureEncoder();

  // Add a buffer to the MTL4 queue's residency set (so MTL4 commands
  // see it as resident). Called from MetalStream after addToResidencySet
  // for the global residency set.
  void addQueueResidency(id<MTLResidencySet> set);

 private:
  id<MTLDevice> device_;       // borrowed
  ResidencyManager* residency_;// borrowed; nullptr if no residency set

  // MTL4 dispatch state (all retained where applicable).
  id<MTL4CommandQueue> mtl4Queue_ = nil;
  id<MTL4CommandAllocator> mtl4Allocator_ = nil;
  id<MTL4CommandBuffer> mtl4CommandBuffer_ = nil;
  id<MTL4CommandBuffer> mtl4InFlightCommandBuffer_ = nil;
  id<MTL4ComputeCommandEncoder> mtl4Encoder_ = nil;
  id<MTL4ArgumentTable> mtl4ArgTable_ = nil;

  // Inline-bytes / scalar bump-scratch buffer (1MB, shared storage).
  id<MTLBuffer> mtl4ScalarScratch_ = nil;
  size_t mtl4ScalarScratchOffset_ = 0;

  // Completion event drained by wait().
  id<MTLSharedEvent> mtl4CompletionEvent_ = nil;
  uint64_t mtl4CompletionValue_ = 0;


  // MTL4 shader compiler (lazy-created on first compile call). Owned
  // here rather than on MetalKernelCompiler so MTL4-specific objects
  // live with the MTL4 backend.
  id<MTL4Compiler> mtl4Compiler_ = nil;
};

#endif  // ET_METAL4_ENABLE

} // namespace metal_v2
} // namespace backends
} // namespace executorch
