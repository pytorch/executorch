/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MpsInterop — MPSGraph interop for the v2 backend.
// MPSGraph requires a legacy id<MTLCommandBuffer>. When ET_METAL_USE_MPSGRAPH
// is on, MetalStream owns an MpsInterop and exposes encodeWithLegacyCommandBuffer
// for ops (currently MPSGraphOp) that need to encode MPSGraph work.
//===----------------------------------------------------------------------===//
// Build-time gate
//===----------------------------------------------------------------------===//
// ET_METAL_USE_MPSGRAPH (default: 1)
//   ON  — compile in MPSGraph support. Requires the MTL3 dispatch path
//         (legacy MTLCommandBuffer model). Must NOT be combined with
//         ET_METAL4_ENABLE.
//   OFF — compile out MPSGraph entirely. MetalStream loses encodeWithLegacy-
//         CommandBuffer / mps() accessor; MPSGraphOp + MPSGraph fallback
//         routing in MatMulOp are dead code; MetalCommandRecorder doesn't
//         own MpsInterop.
// Why mutually exclusive with ET_METAL4_ENABLE:
//   MTL4's command-buffer model is incompatible with the legacy id<MTLCommandBuffer>
//   that MPSGraph targets. The pre-2026 v2 had a cross-queue MTLSharedEvent
//   sync path to bridge them — ~50 LOC of fragile event accounting plus
//   dead virtuals on IComputeBackend. We dropped it because:
//     (a) nobody used MPSGraph + MTL4 together in production
//     (b) the cross-queue sync was a real correctness liability under
//         OS upgrades
//     (c) native op coverage in metal_v2 is good enough that MPSGraph
//         fallback under MTL4 isn't load-bearing
//   If MPSGraph + MTL4 ever becomes a real requirement, the bridge code
//   is in git history (search "MpsBridge" and "commitAndSignal").
//===----------------------------------------------------------------------===//

#ifndef ET_METAL_USE_MPSGRAPH
#define ET_METAL_USE_MPSGRAPH 1
#endif

#ifndef ET_METAL4_ENABLE
#define ET_METAL4_ENABLE 0
#endif

#if ET_METAL_USE_MPSGRAPH && ET_METAL4_ENABLE
#error \
"ET_METAL_USE_MPSGRAPH and ET_METAL4_ENABLE are mutually exclusive. " \
"MPSGraph requires the legacy MTLCommandBuffer model, which is " \
"incompatible with MTL4. Build with one or the other (or neither)."
#endif

#if ET_METAL_USE_MPSGRAPH

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <functional>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalCommandRecorder;

class MpsInterop {
 public:
  // All references are BORROWED. MetalStream constructs the interop and
  // outlives it.
  MpsInterop(MetalCommandRecorder* recorder, id<MTLCommandQueue> queue);
  ~MpsInterop() = default;

  MpsInterop(const MpsInterop&) = delete;
  MpsInterop& operator=(const MpsInterop&) = delete;

  // Encode work that needs a legacy MTLCommandBuffer. Closes the recorder's
  // active compute encoder, hands the legacy CB to encode_fn (which does
  // [graph encodeToCommandBuffer:]), then adopts back any CB that
  // commitAndContinue: may have swapped in. Subsequent typed-setter
  // dispatches encode into the same CB. Single end-of-execute commit.
  void encodeWithLegacyCommandBuffer(
      std::function<void(MPSCommandBuffer* mpsCB)> encode_fn);

 private:
  MetalCommandRecorder* recorder_;  // borrowed
  id<MTLCommandQueue> queue_;       // borrowed (owned by MetalStream)
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch

#endif // ET_METAL_USE_MPSGRAPH
