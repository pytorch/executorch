/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/runtimes/metal/MetalBuffer.h>

#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>

namespace executorch {
namespace backends {
namespace native {

MetalBuffer::~MetalBuffer() {
  if (mode_ == Mode::Owned && ptr_ && stream_) {
    stream_->free(ptr_);
  } else if (mode_ == Mode::NdmAlias) {
    ndm_buffer_.Free();
    // Note: we don't unregister from MetalStream's ptrToBuffer_ here.
    // For Owned mode, stream_->free() handles pool return AND ptrToBuffer_
    // bookkeeping; for NdmAlias, the registration persists for the
    // delegate's lifetime, which is correct since constants are
    // permanent. Net leak per delegate teardown is bounded by the
    // number of constants, and the next delegate gets a fresh stream.
  }
  // Aliasing: nothing to free.
}

void MetalBuffer::re_alias(void* ptr, size_t bytes) {
  // Idempotent: already aliased here.
  if (ptr_ == ptr && size_bytes() == bytes) {
    return;
  }
  // Owned → Aliasing transition: return pool memory to the stream and
  // switch modes so the destructor doesn't double-free.
  if (mode_ == Mode::Owned && ptr_ && stream_) {
    stream_->free(ptr_);
  }
  mode_ = Mode::Aliasing;
  ptr_ = ptr;
  set_size_bytes(bytes);
  // Note: caller is responsible for stream_->registerExternalBuffer(ptr)
  // so dispatches resolve via bufferForPtr().
}

}  // namespace native
}  // namespace backends
}  // namespace executorch
