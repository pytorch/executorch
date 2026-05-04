/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalMTL3Backend — Metal 3 implementation of IComputeBackend.
// Owns:
//   - The legacy MTLCommandBuffer being assembled (commandBuffer_).
//   - The active MTLComputeCommandEncoder (encoder_).
//   - The in-flight MTLCommandBuffer awaiting GPU completion (inFlight_).
// Does NOT own (uses borrowed references):
//   - id<MTLDevice>          — passed in by MetalStream.
//   - id<MTLCommandQueue>    — passed in by MetalStream (also used by MPS
//                              interop directly).
// Lifetime:
//   - One backend instance per MetalStream.
//   - Backend can outlive its in-flight CB; destructor blocks on pending
//     work (in case caller drops the stream without calling wait()).
//===----------------------------------------------------------------------===//

#include <executorch/backends/portable/runtime/metal_v2/IComputeBackend.h>

#import <Metal/Metal.h>

#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalMTL3Backend : public IComputeBackend,
                         public ILegacyCommandBufferProvider {
 public:
  // device + queue are NOT retained by us — caller (MetalStream) owns their
  // lifetime. We do retain command buffers and encoders we create.
  MetalMTL3Backend(id<MTLDevice> device, id<MTLCommandQueue> queue);
  ~MetalMTL3Backend() override;

  MetalMTL3Backend(const MetalMTL3Backend&) = delete;
  MetalMTL3Backend& operator=(const MetalMTL3Backend&) = delete;

  // ---- IComputeBackend ----
  void setKernel(MetalKernel* kernel) override;
  void setBuffer(id<MTLBuffer> buf, size_t offset, uint32_t slot) override;
  void setBytes(const void* ptr, size_t size, uint32_t slot) override;
  void dispatch(uvec3 grid, uvec3 block) override;
  // Concurrent-mode aware. When the encoder is created with
  // MTLDispatchTypeConcurrent (gated by ET_METAL_USE_METAL3_CONCURRENT=1),
  // dispatches in the same encoder run in parallel — we MUST insert a
  // memoryBarrierWithScope:Buffers before any dispatch that depends on a
  // prior pending write. The HazardTracker tells us when via the bool.
  // When the encoder is serial (default), MTL3 auto-serializes dispatches
  // and we can ignore the hint.
  void dispatchHazardAware(uvec3 grid, uvec3 block, bool insertBarrierBefore) override;
  void endEncoder() override;
  void commit() override;
  void wait() override;
  id<MTLCommandBuffer> ensureLegacyCommandBuffer() override;
  void adoptLegacyCommandBuffer(id<MTLCommandBuffer> newCB) override;
  void releaseLegacyCommandBuffer() override;

 private:
  // Lazily create commandBuffer_ from queue_.
  void ensureCommandBuffer();
  // Lazily create encoder_ on commandBuffer_.
  void ensureEncoder();

  id<MTLDevice> device_;          // borrowed
  id<MTLCommandQueue> queue_;     // borrowed

  id<MTLCommandBuffer> commandBuffer_ = nil;       // retained
  id<MTLComputeCommandEncoder> encoder_ = nil;     // retained
  id<MTLCommandBuffer> inFlightCommandBuffer_ = nil; // retained
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
