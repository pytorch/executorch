/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Buffer.h>

#include <executorch/runtime/core/freeable_buffer.h>

#include <cstddef>
#include <utility>

// Forward declaration so MetalBuffer.h stays pure C++ (no Metal/ObjC
// imports). The actual id<MTLBuffer> is owned by the stream's MetalAllocator
// pools (ptrToBuffer_), so MetalBuffer just holds the host_ptr from
// stream->alloc() / registerExternalBuffer() and the byte count.
namespace executorch {
namespace backends {
namespace metal_v2 {
class MetalStream;
} // namespace metal_v2
} // namespace backends
} // namespace executorch

namespace executorch {
namespace backends {
namespace native {

/**
 * Concrete Buffer subclass for Apple Silicon Metal storage.
 *
 * On Apple Silicon, MTLBuffer.contents is a host-addressable pointer
 * (unified memory). We therefore store the host pointer here directly;
 * the underlying id<MTLBuffer> lives in the stream's MetalAllocator
 * ptrToBuffer_ map, looked up via stream->bufferForPtr() when ops need
 * to encode dispatches against it. This keeps MetalBuffer.h pure-C++
 * and avoids dragging Metal/ObjC headers into runtime_v2 callers.
 *
 * Three lifetime modes (selected at construction):
 *   1. OWNED — wraps memory we got from stream->alloc(); destructor
 *      calls stream->free(ptr_) to return it to the pool.
 *   2. ALIASING — wraps a host pointer the caller keeps alive (router's
 *      cross-runtime alias optimization, or upload_from_host re-aliasing
 *      an Owned-mode Buffer to point at caller storage). Destructor is
 *      a no-op; pool memory was already returned during the Owned →
 *      Aliasing transition by re_alias.
 *   3. NDM_ALIAS — wraps a FreeableBuffer from the NamedDataMap (held
 *      internally; released in destructor). The data was registered with
 *      stream->allocator().registerExternalBuffer() at upload time.
 *
 * The "location" is set per Runtime (router stamps it during routing).
 */
class MetalBuffer : public Buffer {
 public:
  enum class Mode : uint8_t { Owned, Aliasing, NdmAlias };

  // Mode::Owned — `ptr` was returned by stream->alloc(bytes). Destructor
  // calls stream->free(ptr).
  static MetalBuffer* allocate(
      ::executorch::backends::metal_v2::MetalStream* stream,
      void* ptr,
      size_t bytes,
      MemoryKind kind = MemoryKind::DeviceOnly) {
    return new MetalBuffer(Mode::Owned, stream, ptr, bytes, kind);
  }

  // Mode::Aliasing — `ptr` came from caller storage; was registered with
  // the stream by upload_from_host (or set up directly by the caller).
  // Destructor is a no-op (caller keeps host memory alive).
  static MetalBuffer* alias(
      ::executorch::backends::metal_v2::MetalStream* stream,
      void* ptr,
      size_t bytes,
      MemoryKind kind = MemoryKind::DeviceMirror) {
    return new MetalBuffer(Mode::Aliasing, stream, ptr, bytes, kind);
  }

  // Mode::NdmAlias — `fb` is the NDM FreeableBuffer; `ptr` = fb.data().
  // Was registered with the stream via registerExternalBuffer.
  // Destructor releases the FreeableBuffer (which keeps the mmap'd
  // region alive).
  static MetalBuffer* alias_ndm(
      ::executorch::backends::metal_v2::MetalStream* stream,
      ::executorch::runtime::FreeableBuffer&& fb,
      MemoryKind kind = MemoryKind::DeviceOnly) {
    void* ptr = const_cast<void*>(fb.data());
    size_t bytes = fb.size();
    auto* mb = new MetalBuffer(Mode::NdmAlias, stream, ptr, bytes, kind);
    mb->ndm_buffer_.~FreeableBuffer();
    new (&mb->ndm_buffer_)::executorch::runtime::FreeableBuffer(std::move(fb));
    return mb;
  }

  ~MetalBuffer() override; // defined in MetalBuffer.mm to call stream_->allocator().free()

  void* host_ptr() override {
    return ptr_;
  }

  // Re-target this buffer's pointer/size in place.
  //
  // - Idempotent: if `ptr == ptr_` and `bytes == size_bytes()`, no-op.
  // - Owned → Aliasing transition: if currently Owned, returns the
  //   pool-allocated ptr_ to the stream's pool and switches mode to
  //   Aliasing. The new caller-supplied ptr is then stored. The caller
  //   is responsible for stream->registerExternalBuffer(new_ptr) so
  //   dispatches resolve.
  // - Aliasing-mode rebind: just updates ptr/size.
  //
  // Defined in MetalBuffer.mm to call stream_->allocator().free().
  void re_alias(void* ptr, size_t bytes);

  Mode mode() const {
    return mode_;
  }

  ::executorch::backends::metal_v2::MetalStream* stream() const {
    return stream_;
  }

 private:
  MetalBuffer(
      Mode m,
      ::executorch::backends::metal_v2::MetalStream* stream,
      void* ptr,
      size_t bytes,
      MemoryKind kind)
      : Buffer(bytes, kind),
        mode_(m),
        stream_(stream),
        ptr_(ptr),
        ndm_buffer_(nullptr, 0, nullptr, nullptr) {}

  Mode mode_;
  ::executorch::backends::metal_v2::MetalStream* stream_;
  void* ptr_;
  ::executorch::runtime::FreeableBuffer ndm_buffer_;
};

} // namespace native
} // namespace backends
} // namespace executorch
