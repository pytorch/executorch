/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MpsInterop — MPSGraph interop.
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
//
// MPSGraph and MTL4 are mutually exclusive: MTL4's command-buffer model is
// incompatible with the legacy id<MTLCommandBuffer> that MPSGraph targets.
//===----------------------------------------------------------------------===//

#ifndef ET_METAL_USE_MPSGRAPH
#define ET_METAL_USE_MPSGRAPH 1
#endif

#include <executorch/backends/metal/core/MetalConfig.h>

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

#include <cstddef>
#include <functional>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalCommandRecorder;

class MpsInterop {
 public:
  // All references are BORROWED. MetalStream constructs the interop and
  // outlives it.
  MpsInterop(MetalCommandRecorder* recorder);
  ~MpsInterop() = default;

  MpsInterop(const MpsInterop&) = delete;
  MpsInterop& operator=(const MpsInterop&) = delete;

  // Encode work that needs a legacy MTLCommandBuffer. Closes the recorder's
  // active compute encoder, hands the legacy CB to encode_fn (which does
  // [graph encodeToCommandBuffer:]), then adopts back any CB that
  // commitAndContinue: may have swapped in. Subsequent typed-setter
  // dispatches encode into the same CB. Single end-of-execute commit.
  //
  // Side-door binds: MPSGraph encodes its own
  // setBuffer: calls bypassing the recorder's typed-setter API. The
  // caller MUST pass every MTLBuffer it intends MPSGraph to bind via
  // `side_door_binds` so the recorder records them in its per-CB binds_
  // vector and pinBatch covers them at flush time. This is the side-
  // door encoder contract.
  //
  // Failure mode if violated:
  //   MTL4: page fault on missing residency → undefined GPU behavior.
  //         (Not currently triggered: ET_METAL_USE_MPSGRAPH and
  //         ET_METAL4_ENABLE are mutually exclusive, so MPSGraph runs
  //         only on MTL3 today. The wiring is for forward compatibility
  //         if that ever changes.)
  //   MTL3: silent first-touch lazy paging → perf cliff, no correctness.
  //
  // The default empty span is safe for callers that don't bind any
  // MTLBuffers via the side-door encoder (rare; today's MPSGraphOp
  // always binds via feeds/results).
  // Public API takes raw (pointer, count) for side_door_binds instead
  // of std::span to avoid imposing C++20 on every header consumer.
  // Pass nullptr/0 if no buffers need declaration.
  void encodeWithLegacyCommandBuffer(
      std::function<void(MPSCommandBuffer* mpsCB)> encode_fn,
      id<MTLBuffer> const __unsafe_unretained* side_door_binds = nullptr,
      size_t side_door_binds_count = 0);

 private:
  MetalCommandRecorder* recorder_;  // borrowed
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch

#endif // ET_METAL_USE_MPSGRAPH
