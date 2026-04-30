/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MpsInterop.h"

// MpsInterop.h's compile-time gate fully removes the class declaration when
// ET_METAL_USE_MPSGRAPH=0; the rest of this file is then dead code. Wrap
// the whole .mm body in the same gate so the TU compiles to an empty
// object file in that build mode.
#import "MpsInterop.h"

#if ET_METAL_USE_MPSGRAPH

#import "MetalCommandRecorder.h"
#import "IComputeBackend.h"
#import "MetalMTL3Backend.h"  // for ILegacyCommandBufferProvider downcast

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

MpsInterop::MpsInterop(MetalCommandRecorder* recorder, id<MTLCommandQueue> queue)
    : recorder_(recorder), queue_(queue) {}

void MpsInterop::encodeWithLegacyCommandBuffer(
    std::function<void(MPSCommandBuffer* mpsCB)> encode_fn) {
  // wrap the whole body in @autoreleasepool so the autoreleased
  // [queue commandBuffer] / [MPSCommandBuffer commandBufferWithCommandBuffer:]
  // objects drain at scope exit instead of accumulating in an outer pool
  // (or never draining for long-running execute loops).
  @autoreleasepool {
    // Close any open compute encoder so MPS work executes after any prior
    // typed-setter dispatches.
    recorder_->endEncoder();

    // ── MTL3 path: adopt-and-share single legacy cb ──
    // Recorder.mtl3Backend() is a MetalMTL3Backend* (concrete); we reach
    // its ILegacyCommandBufferProvider face via dynamic_cast. Always
    // succeeds because under ET_METAL_USE_MPSGRAPH the only backend in
    // play is MTL3.
    auto* mtl3 = dynamic_cast<ILegacyCommandBufferProvider*>(
        recorder_->mtl3Backend());
    ET_CHECK_MSG(mtl3 != nullptr,
                 "MpsInterop: mtl3Backend() does not implement "
                 "ILegacyCommandBufferProvider — MPSGraph requires MTL3");
    id<MTLCommandBuffer> cb = mtl3->ensureLegacyCommandBuffer();
    MPSCommandBuffer* mpsCB =
        [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
    encode_fn(mpsCB);
    mtl3->adoptLegacyCommandBuffer(mpsCB.commandBuffer);
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch

#endif // ET_METAL_USE_MPSGRAPH
