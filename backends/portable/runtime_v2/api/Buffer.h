/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Location.h>

#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Opaque storage handle. Lives on a Location. Concrete subclasses
 * (HostBuffer, MetalBuffer, VulkanBuffer) are private to each runtime;
 * the router and executor see only Buffer*.
 *
 * Ownership: Buffers are owned by their Provider's RuntimeContext (its
 * pool). Instance::allocate returns a non-owning Buffer*. The Plan records
 * which Instance allocated which Buffer; on destruction asks each Instance
 * to release.
 *
 * Only host_ptr() is virtual; location() and size_bytes() read base
 * member storage.
 *
 * Note: lifetime of underlying storage (e.g., a held FreeableBuffer for
 * NDM-aliased constants, or a pool slot) is the concrete subclass's
 * responsibility.
 */
class Buffer {
 public:
  virtual ~Buffer() = default;

  Location location() const { return location_; }
  size_t size_bytes() const { return size_bytes_; }

  // Non-null iff host-addressable. CPU: always. Metal on Apple Silicon:
  // usually (MTLStorageModeShared). Discrete GPU: nullptr.
  virtual void* host_ptr() { return nullptr; }

 protected:
  Buffer(Location loc, size_t bytes) : location_(loc), size_bytes_(bytes) {}

  // Allow derived classes (e.g., recycled HostBuffer slots) to re-set size
  // when re-aliasing the underlying storage.
  void set_size_bytes(size_t bytes) { size_bytes_ = bytes; }

private:
  Location location_;
  size_t size_bytes_;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
