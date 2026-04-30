/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/MemoryKind.h>

#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace native {

/**
 * Opaque storage handle. Concrete subclasses (HostBuffer, MetalBuffer,
 * VulkanBuffer) are private to each runtime; the router and executor
 * see only Buffer*.
 *
 * Ownership: Buffers are owned by their Runtime's RuntimeContext (its
 * pool). Engine::allocate returns a non-owning Buffer*. The Plan records
 * which Engine allocated which Buffer (Plan::value_owner); on destruction
 * asks each Engine to release.
 *
 * Each Buffer carries a MemoryKind (the addressing contract under which
 * it was allocated). The kind is set at construction by the backend in
 * response to the AllocRequest::kind it was asked to satisfy and is
 * stable for the buffer's lifetime.
 *
 * Only host_ptr() is virtual; size_bytes() and memory_kind() read base
 * member storage.
 *
 * Note: lifetime of underlying storage (e.g., a held FreeableBuffer for
 * NDM-aliased constants, or a pool slot) is the concrete subclass's
 * responsibility.
 *
 * Physical addressability is derivable: HostOnly/HostMirror live in
 * host RAM; DeviceMirror/DeviceOnly live in the owning Engine's
 * runtime address space (consult Plan::value_owner if needed).
 */
class Buffer {
 public:
  virtual ~Buffer() = default;

  size_t size_bytes() const {
    return size_bytes_;
  }
  MemoryKind memory_kind() const {
    return kind_;
  }

  // Non-null iff host-addressable. Runtime-defined per allocation:
  //  - HostOnly / HostMirror: always non-null.
  //  - DeviceMirror: non-null on UMA (where the pair is collapsed) or
  //    when the provider deliberately exposes the device side as host
  //    addressable; nullptr on discrete GPU pairs (caller must use
  //    upload_from_host / download_to_host).
  //  - DeviceOnly: provider-defined; nullptr is the conservative
  //    default for discrete GPU.
  virtual void* host_ptr() {
    return nullptr;
  }

 protected:
  Buffer(size_t bytes, MemoryKind kind = MemoryKind::HostOnly)
      : size_bytes_(bytes), kind_(kind) {}

  // Allow derived classes (e.g., recycled HostBuffer slots) to re-set size
  // when re-aliasing the underlying storage.
  void set_size_bytes(size_t bytes) {
    size_bytes_ = bytes;
  }

 private:
  size_t size_bytes_;
  MemoryKind kind_;
};

} // namespace native
} // namespace backends
} // namespace executorch
