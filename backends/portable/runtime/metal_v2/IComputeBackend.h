/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// IComputeBackend — virtual interface for the per-dispatch encoder API.
// Two implementations exist (one is selected at MetalStream construction):
//   - MetalMTL3Backend  : MTLCommandQueue + MTLCommandBuffer +
//                         MTLComputeCommandEncoder (the legacy, default path).
//   - MetalMTL4Backend  : MTL4CommandQueue + MTL4CommandBuffer +
//                         MTL4ComputeCommandEncoder + MTL4ArgumentTable
//                         (Metal 4 path, requires SDK 26+ and runtime
//                         @available(macOS 26.0, *)).
// Why this interface exists:
//   * Before R6, MetalStream interleaved MTL3 + MTL4 dispatch logic with
//     #if ET_METAL4_ENABLE blocks across many sites. This was the source of
//     several intermittent bugs (arg-table snapshot mismatch, fence-index
//     gaps, scalar-scratch alignment).
//   * After R6, all backend-specific code lives in the corresponding
//     subclass file. MetalStream calls this interface and is unaware of
//     which backend it owns.
// Design constraints:
//   * The interface is INTENTIONALLY NARROW. Anything richer belongs on
//     MetalStream (which owns BufferRegistry, KernelCatalog, MPSGraph
//     bridge — these are cross-backend concerns).
//   * setBuffer/setBytes/setKernel/dispatch capture state at the moment
//     of the call. Callers can free pointers immediately after dispatch()
//     returns. (MTL3 setBytes copies into the encoder's command stream;
//     MTL4 setBytes copies into a backend-private bump scratch.)
//   * commit() is non-blocking. wait() blocks until all work previously
//     committed has finished GPU execution.
//===----------------------------------------------------------------------===//

#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>

#import <Metal/Metal.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Forward decl — backends pass MetalKernel through opaquely; the resolved
// MTLComputePipelineState is fetched via metalKernel->pipeline() inside the
// backend.
class MetalKernel;

class IComputeBackend {
 public:
  virtual ~IComputeBackend() = default;

  //===--------------------------------------------------------------------===//
  // Per-dispatch encoding
  // Logical sequence per dispatch:
  //   backend->setKernel(metalKernel);
  //   backend->setBuffer(buf, offset, slot=0);
  //   backend->setBytes(&scalar, sizeof(scalar), slot=1);
  //   ...
  //   backend->dispatch(grid, block);
  // The backend is responsible for opening an encoder lazily on the first
  // setKernel call after a previous endEncoder()/commit().
  //===--------------------------------------------------------------------===//

  // Bind the compute pipeline for the next dispatch. The MetalKernel*
  // points to a cached PSO owned by MetalKernelCompiler (which lives on
  // MetalStream — outside the backend's concern).
  virtual void setKernel(MetalKernel* kernel) = 0;

  // Bind a buffer at the given argument slot. `offset` is in bytes from
  // the start of the MTLBuffer. The MTLBuffer is owned by MetalStream's
  // BufferRegistry; the backend just records the binding.
  virtual void setBuffer(id<MTLBuffer> buf, size_t offset, uint32_t slot) = 0;

  // Bind small inline data at the given slot. Bytes are SNAPSHOTTED at
  // call time — caller's pointer can die immediately after this returns.
  // Cap is kMaxInlineBytes (4 KiB) per call.
  // MTL3 backend: routes to [encoder setBytes:length:atIndex:].
  // MTL4 backend: bump-allocates from mtl4ScalarScratch_, writes bytes,
  //               binds the address via [argTable setAddress:atIndex:].
  virtual void setBytes(const void* ptr, size_t size, uint32_t slot) = 0;

  // Dispatch the kernel previously bound via setKernel + setBuffer/setBytes.
  // Threadgroup dimensions are fixed by the kernel; grid is in threadgroups.
  virtual void dispatch(uvec3 grid, uvec3 block) = 0;

  //===--------------------------------------------------------------------===//
  // R8.1: Hazard-aware dispatch (used by typed-setter path on MetalStream).
  // Same semantics as dispatch() above EXCEPT the caller has analyzed
  // read/write hazards with prior pending dispatches and tells us whether
  // a memory barrier is needed BEFORE this dispatch.
  //   - insertBarrierBefore=true: insert a barrier so this dispatch's
  //     reads/writes synchronize with prior dispatches' writes. Required
  //     when there's a RAW or WAW hazard.
  //   - insertBarrierBefore=false: no barrier — caller has proven this
  //     dispatch is independent from all prior pending dispatches.
  //     UNSAFE if untrue.
  // MTL3 backend: ignores the flag (Metal 3 has automatic hazard tracking
  //               within a compute encoder).
  // MTL4 backend: honors the flag (Metal 4 has NO automatic hazard tracking;
  //               without explicit barriers, dispatches in the same encoder
  //               run concurrently and silently produce wrong numerics).
  virtual void dispatchHazardAware(
      uvec3 grid, uvec3 block, bool insertBarrierBefore) = 0;

  //===--------------------------------------------------------------------===//
  // Encoder lifecycle
  // The backend opens an encoder lazily; endEncoder() closes it. Multiple
  // dispatches between endEncoder() calls share one encoder (Metal serializes
  // dispatches within an encoder; backends may insert per-dispatch barriers
  // to enforce ordering depending on their model).
  //===--------------------------------------------------------------------===//

  // Close any open compute encoder. Safe to call when no encoder is open.
  virtual void endEncoder() = 0;

  //===--------------------------------------------------------------------===//
  // Submission
  // commit() finalizes the current command buffer (closes the encoder if open,
  // submits to the GPU queue, signals completion event). Non-blocking — the
  // committed work runs asynchronously. Stores the in-flight handle internally
  // so wait() can drain it later.
  // wait() blocks until all previously-committed work has GPU-completed. Safe
  // to call when nothing is in flight (no-op).
  //===--------------------------------------------------------------------===//

  virtual void commit() = 0;
  virtual void wait() = 0;
};

//===----------------------------------------------------------------------===//
// ILegacyCommandBufferProvider — MTL3-specific extension. Only implemented
// by MetalMTL3Backend; only used by MpsInterop's MTL3 path. Kept off the
// universal IComputeBackend interface because MTL4 has no legacy-CB concept
// (and the MPSGraph + MTL4 combo is build-flag mutually exclusive — see
// MpsInterop.h).
//
// MpsInterop's MTL3 path holds an ILegacyCommandBufferProvider*; under MTL4
// builds (ET_METAL_USE_MPSGRAPH=0) MpsInterop itself is compiled out and
// nobody references this interface.
//===----------------------------------------------------------------------===//

class ILegacyCommandBufferProvider {
 public:
  virtual ~ILegacyCommandBufferProvider() = default;

  // Ensure a legacy MTLCommandBuffer exists; return it. Sets hasPendingWork
  // on the underlying state.
  virtual id<MTLCommandBuffer> ensureLegacyCommandBuffer() = 0;
  // Replace the current legacy CB with a new one (typically after MPS
  // commitAndContinue returns a fresh CB).
  virtual void adoptLegacyCommandBuffer(id<MTLCommandBuffer> newCB) = 0;
  // Release the current legacy CB without committing (used by MPS interop
  // after MPSCommandBuffer takes ownership).
  virtual void releaseLegacyCommandBuffer() = 0;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
