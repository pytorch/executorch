/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Buffer.h>

#include <executorch/runtime/core/freeable_buffer.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Concrete Buffer subclass for host (CPU) memory.
 *
 * Three lifetime modes (selected at construction):
 *   1. OWNED — wraps memory we malloc'd; freed in destructor.
 *   2. ALIASING — wraps a host pointer the caller keeps alive (used by
 *      HostImportArena and bind_inputs/bind_outputs); destructor is a no-op.
 *   3. NDM_ALIAS — wraps a NamedDataMap FreeableBuffer (held internally,
 *      released in destructor).
 *
 * The arena/recycler can re-target an ALIASING HostBuffer's pointer/size
 * via re_alias() to recycle slots.
 */
class HostBuffer : public Buffer {
 public:
  enum class Mode : uint8_t { Owned, Aliasing, NdmAlias };

  // Mode::Owned — allocate `bytes` via std::aligned_alloc.
  static HostBuffer* allocate(size_t bytes, size_t alignment) {
    void* mem = std::aligned_alloc(
        alignment, ((bytes + alignment - 1) / alignment) * alignment);
    return new HostBuffer(Mode::Owned, mem, bytes);
  }

  // Mode::Aliasing — wrap a caller-owned pointer.
  static HostBuffer* alias(void* ptr, size_t bytes) {
    return new HostBuffer(Mode::Aliasing, ptr, bytes);
  }

  // Mode::NdmAlias — wrap a FreeableBuffer (move-in; held until ~HostBuffer).
  static HostBuffer* alias_ndm(::executorch::runtime::FreeableBuffer&& fb) {
    void* ptr = const_cast<void*>(fb.data());
    size_t bytes = fb.size();
    auto* hb = new HostBuffer(Mode::NdmAlias, ptr, bytes);
    hb->ndm_buffer_.~FreeableBuffer();
    new (&hb->ndm_buffer_) ::executorch::runtime::FreeableBuffer(std::move(fb));
    return hb;
  }

  ~HostBuffer() override {
    if (mode_ == Mode::Owned && ptr_) {
      std::free(ptr_);
    } else if (mode_ == Mode::NdmAlias) {
      ndm_buffer_.Free();
    }
    // Aliasing: nothing to do.
  }

  void* host_ptr() override { return ptr_; }

  // Re-target this buffer's pointer/size in place.
  //
  // - Idempotent: if `ptr == ptr_` and `bytes == size_bytes()`, no-op.
  // - Owned → Aliasing transition: if currently Owned and ptr_ != nullptr,
  //   frees the malloc'd storage (so the destructor doesn't double-free)
  //   and switches mode to Aliasing.
  // - Aliasing-mode rebind: just updates ptr/size.
  // - NdmAlias is invalid here (NDM buffers shouldn't be re-aliased);
  //   asserts in debug.
  //
  // Used by HostImportArena AND by upload_from_host's rebind path so
  // upload-time aliasing replaces a stale Owned allocation in place.
  void re_alias(void* ptr, size_t bytes) {
    if (ptr_ == ptr && size_bytes() == bytes) {
      return;  // already aliased here; cheap idempotent path
    }
    if (mode_ == Mode::Owned && ptr_) {
      std::free(ptr_);
    }
    mode_ = Mode::Aliasing;
    ptr_ = ptr;
    set_size_bytes(bytes);
  }

  Mode mode() const { return mode_; }

 private:
  HostBuffer(Mode m, void* ptr, size_t bytes)
      : Buffer(Location::host(), bytes),
        mode_(m),
        ptr_(ptr),
        ndm_buffer_(nullptr, 0, nullptr, nullptr) {}

  Mode mode_;
  void* ptr_;
  ::executorch::runtime::FreeableBuffer ndm_buffer_;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
