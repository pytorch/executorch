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
#if ET_METAL_USE_MPSGRAPH

#import "MetalCommandRecorder.h"
#import "MetalMTL3Backend.h"
#import "IComputeBackend.h"

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

MpsInterop::MpsInterop(MetalCommandRecorder* recorder)
    : recorder_(recorder) {}

void MpsInterop::encodeWithLegacyCommandBuffer(
    std::function<void(MPSCommandBuffer* mpsCB)> encode_fn,
    id<MTLBuffer> const __unsafe_unretained* side_door_binds,
    size_t side_door_binds_count) {
  // wrap the whole body in @autoreleasepool so the autoreleased
  // [queue commandBuffer] / [MPSCommandBuffer commandBufferWithCommandBuffer:]
  // objects drain at scope exit instead of accumulating in an outer pool
  // (or never draining for long-running execute loops).
  @autoreleasepool {
    // Side-door encoder contract.
    // Declare the buffers MPSGraph will bind via setBuffer: BEFORE the
    // encode_fn runs. The recorder records them in its per-CB binds_
    // vector; pinBatch at flush() will cover them. We do this BEFORE
    // endEncoder() / encode_fn so the binds are present in the same
    // CB that MPSGraph encodes into.
    if (side_door_binds && side_door_binds_count > 0) {
      recorder_->declareSideDoorBinds(side_door_binds, side_door_binds_count);
    }

    // Close any open compute encoder so MPS work executes after any prior
    // typed-setter dispatches.
    recorder_->endEncoder();

    // ── MTL3 path: adopt-and-share single legacy cb ──
    // Reach ILegacyCommandBufferProvider via the IComputeBackend virtual
    // hook. Always non-null under ET_METAL_USE_MPSGRAPH because the only
    // backend in play is MTL3.
    auto* mtl3 = recorder_->mtl3Backend()->legacyCommandBufferProvider();
    ET_CHECK_MSG(mtl3 != nullptr,
                 "MpsInterop: mtl3Backend() does not provide "
                 "ILegacyCommandBufferProvider — MPSGraph requires MTL3");
    id<MTLCommandBuffer> cb = mtl3->ensureLegacyCommandBuffer();
    MPSCommandBuffer* mpsCB =
        [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
    encode_fn(mpsCB);
    mtl3->adoptLegacyCommandBuffer(mpsCB.commandBuffer);
    // Mark the recorder as having pending work so the next flush()
    // commits the legacy CB. Without this, an MPSGraph-only execution
    // (no preceding/following typed-setter dispatch) would leave the
    // CB unsubmitted and the MPS work would never reach the GPU.
    recorder_->noteMpsWorkPending();
  }
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch

#endif // ET_METAL_USE_MPSGRAPH
