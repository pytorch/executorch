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
  // Only the host runtime addresses this buffer. Runtime chooses the
  // allocator (e.g., aligned malloc, hugepages).
  HostOnly = 0,

  // Host side of a host↔device mirror pair. Both host code and the
  // paired device runtime see this value, accessed via the pair. The
  // partner DeviceMirror may share storage with this buffer (UMA
  // collapse) or be a distinct physical allocation linked by
  // per-execute copies.
  HostMirror = 1,

  // Device side of a host↔device mirror pair. Symmetric counterpart of
  // HostMirror.
  DeviceMirror = 2,

  // Only the owning provider's runtime addresses this buffer. Runtime
  // chooses the allocator (e.g., cudaMalloc for a GPU intermediate;
  // cudaMallocHost for a KV cache the GPU reads via DMA without
  // staging; MTLStorageModePrivate or MTLStorageModeShared on Metal).
  // The host has no contract guarantee that host_ptr() is non-null;
  // backends MAY still expose one for diagnostics on UMA platforms.
  DeviceOnly = 3,
};

inline const char* to_string(MemoryKind k) {
  switch (k) {
    case MemoryKind::HostOnly:
      return "HostOnly";
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
