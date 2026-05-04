/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalKernel - Compiled Metal compute pipeline + per-slot access metadata
// Wraps an id<MTLComputePipelineState>, the kernel's MSL host_name, and a
// (slot → SlotAccess) map populated by MetalKernelCompiler at compile time
// from the parsed MSL source. The slot map drives the AOTI dispatcher's
// routing of args through setInput / setOutput / setInOut.
// Extracted from MetalStream.h — definition unchanged.
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalKernel {
public:
  // R8.4: Per-slot access mode parsed from the kernel's MSL source. Used
  // by callers (specifically the AOTI dispatcher) to route each buffer
  // arg through setInput / setOutput / setInOut precisely. For metal_v2
  // hand-written ops, the call site already knows the role and uses the
  // typed setter directly — these slots may be Unknown.
  enum class SlotAccess : uint8_t {
    Unknown,    // not a buffer slot, or not parsed (default)
    ReadOnly,   // `device const T*` or `constant T*`  → setInput
    WriteOnly,  // (rare; MSL [[access(write)]])       → setOutput
    ReadWrite,  // `device T*` (no const)              → setInOut
  };

  MetalKernel(id<MTLComputePipelineState> pipeline, const char* name);
  ~MetalKernel();

  const char* name() const { return name_.c_str(); }
  uvec3 maxThreadsPerThreadgroup() const;
  void* nativeHandle() { return (__bridge void*)pipeline_; }

  id<MTLComputePipelineState> pipeline() const { return pipeline_; }

  // R8.4: install per-slot access modes parsed from the kernel source.
  // Populated by MetalKernelCompiler immediately after construction.
  // Indexed by buffer slot number; sparse (slots not present default to
  // Unknown).
  void setSlotAccess(uint32_t slot, SlotAccess access);

  // R8.4: query the access mode for `slot`. Returns Unknown if the slot
  // wasn't recorded (kernel source not parsed, or slot is a scalar).
  SlotAccess accessForSlot(uint32_t slot) const;

private:
  id<MTLComputePipelineState> pipeline_;
  std::string name_;
  // R8.4: slot → access. Keyed by slot index; small (typically <= 12).
  std::unordered_map<uint32_t, SlotAccess> slotAccess_;
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
