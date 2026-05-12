/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch {
namespace backends {
namespace native {

/**
 * Addressing contract for a Buffer.
 *
 * MemoryKind expresses *who can address* the buffer. It does NOT
 * prescribe how the provider allocates the underlying memory — that is
 * the provider's internal decision and may vary per allocation
 * (cudaMalloc vs cudaMallocHost vs cudaMallocManaged on CUDA;
 * MTLStorageModeShared vs MTLStorageModePrivate on Metal; plain malloc
 * vs page-locked on the host pool).
 *
 * Mirror pairs:
 *   A logical "host↔device mirror" is two AllocRequests emitted
 *   together by the router: one HostMirror on the host pool, one
 *   DeviceMirror on the device runtime, paired via
 *   AllocRequest::mirror_partner. The provider may collapse the pair
 *   into a single physical allocation (e.g., MTLStorageModeShared on
 *   Apple Silicon UMA) or back it with two physical allocations and
 *   per-execute copies (e.g., cudaMallocHost + cudaMalloc on a
 *   discrete GPU). The graph topology never sees the chosen strategy;
 *   it only sees the contract.
 *
 * Sync values are NOT a MemoryKind. They live in Plan::events as
 * EventSlots, completely separate from buffers.
 */
enum class MemoryKind : uint8_t {
  // Caller-owned host storage, wrapped per-execute via bind_inputs /
  // bind_outputs. Used for graph IO. Not biddable: only HostPool
  // handles it (allocates a thin Aliasing HostBuffer; the underlying
  // bytes belong to the caller). No arena slot.
  HostExtern = 0,

  // Host side of a host↔device mirror pair. Delegate-allocated at init
  // (arena-backed). Biddable: any engine may claim (e.g., a CUDA engine
  // could allocate as cudaMallocHost for pinning); HostPool floors. The
  // partner DeviceMirror may share storage (UMA collapse) or be a
  // distinct physical allocation linked by per-execute copies.
  HostMirror = 1,

  // Device side of a host↔device mirror pair. Symmetric counterpart of
  // HostMirror. Single allocator: the targeted device engine.
  DeviceMirror = 2,

  // Only the owning provider's runtime addresses this buffer. Single
  // allocator: the targeted device engine. Runtime chooses the
  // allocator (e.g., cudaMalloc, MTLStorageModePrivate). The host has
  // no contract guarantee that host_ptr() is non-null; backends MAY
  // still expose one for diagnostics on UMA platforms.
  DeviceOnly = 3,
};

inline const char* to_string(MemoryKind k) {
  switch (k) {
    case MemoryKind::HostExtern:
      return "HostExtern";
    case MemoryKind::HostMirror:
      return "HostMirror";
    case MemoryKind::DeviceMirror:
      return "DeviceMirror";
    case MemoryKind::DeviceOnly:
      return "DeviceOnly";
  }
  return "?";
}

} // namespace native
} // namespace backends
} // namespace executorch
