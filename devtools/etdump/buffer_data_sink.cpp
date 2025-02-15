/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/buffer_data_sink.h>
#include <executorch/devtools/etdump/utils.h>

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace executorch {
namespace etdump {

Result<size_t> BufferDataSink::write(const void* ptr, size_t length) {
  if (length == 0) {
    return static_cast<size_t>(-1);
  }
  uint8_t* offset_ptr =
      internal::alignPointer(debug_buffer_.data() + offset_, 64);

  // Zero out the padding between data blobs.
  size_t n_zero_pad = offset_ptr - debug_buffer_.data() - offset_;
  memset(debug_buffer_.data() + offset_, 0, n_zero_pad);

  offset_ = (offset_ptr - debug_buffer_.data()) + length;

  // Raise access error if offset_ is out of range.
  if (offset_ > debug_buffer_.size()) {
    return Error::AccessFailed;
  }

  memcpy(offset_ptr, ptr, length);
  return (size_t)(offset_ptr - debug_buffer_.data());
}

Result<size_t> BufferDataSink::get_storage_size() const {
  return debug_buffer_.size();
}

size_t BufferDataSink::get_used_bytes() const {
  return offset_;
}

} // namespace etdump
} // namespace executorch
