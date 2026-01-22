/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/data_sinks/buffer_data_sink.h>
#include <executorch/devtools/etdump/utils.h>

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

namespace executorch {
namespace etdump {

Result<BufferDataSink> BufferDataSink::create(
    Span<uint8_t> buffer,
    size_t alignment) noexcept {
  // Check if alignment is a power of two and greater than 0
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Error::InvalidArgument;
  }

  return BufferDataSink(buffer, alignment);
}

Result<BufferDataSink>
BufferDataSink::create(void* ptr, size_t size, size_t alignment) noexcept {
  return BufferDataSink::create({(uint8_t*)ptr, size}, alignment);
}

Result<size_t> BufferDataSink::write(const void* ptr, size_t length) {
  bool inPlaceTensor = false;

  if (length != 0 && ptr == nullptr) {
    inPlaceTensor = true;
  } else if (length == 0 || ptr == nullptr) {
    ET_LOG(Info, "Invalid data to write to buffer");
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

  if (inPlaceTensor) {
    memset(cur_data_begin, 0, length);
  } else {
    memcpy(cur_data_begin, ptr, length);
  }

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
