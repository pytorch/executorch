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
    return offset_;
  }

  uint8_t* last_data_end = debug_buffer_.data() + offset_;

  // The beginning of the next data blob must be aligned to the alignment
  uint8_t* cur_data_begin = internal::align_pointer(last_data_end, alignment_);
  uint8_t* cur_data_end = cur_data_begin + length;

  if (cur_data_end > debug_buffer_.data() + debug_buffer_.size()) {
    ET_LOG(Error, "Ran out of space to store intermediate outputs.");
    return Error::OutOfResources;
  }

  // Zero out the padding between data blobs
  memset(last_data_end, 0, cur_data_begin - last_data_end);
  memcpy(cur_data_begin, ptr, length);
  offset_ = (size_t)(cur_data_end - debug_buffer_.data());

  return (size_t)(cur_data_begin - debug_buffer_.data());
}

Result<size_t> BufferDataSink::get_storage_size() const {
  return debug_buffer_.size();
}

size_t BufferDataSink::get_used_bytes() const {
  return offset_;
}

} // namespace etdump
} // namespace executorch
