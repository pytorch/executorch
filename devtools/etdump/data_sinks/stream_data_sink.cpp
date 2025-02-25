/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/data_sinks/stream_data_sink.h>
#include <executorch/devtools/etdump/utils.h>
#include <fcntl.h> // open()
#include <unistd.h> // For write and close
#include <cstring>

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace executorch {
namespace etdump {

StreamDataSink::StreamDataSink(StreamDataSink&& other) noexcept
    : buffer_(std::move(other.buffer_)),
      buffer_offset_(other.buffer_offset_),
      file_descriptor_(other.file_descriptor_),
      total_written_bytes_(other.total_written_bytes_),
      alignment_(other.alignment_) {
  other.file_descriptor_ = -1;
}

Result<StreamDataSink> StreamDataSink::create(
    void* buffer_data_ptr,
    size_t buffer_size,
    const char* file_path,
    size_t alignment) {
  // Check if alignment is a power of two
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Error::InvalidArgument;
  }

  // Open the file and get the file descriptor
  // It will fail if the file already exists
  int file_descriptor = open(file_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (file_descriptor < 0) {
    // Return an error if the file cannot be accessed or created
    ET_LOG(
        Error, "Failed to open %s: %s (%d)", file_path, strerror(errno), errno);
    return Error::AccessFailed;
  }

  // Return the successfully created StreamDataSink
  return StreamDataSink(
      buffer_data_ptr, buffer_size, file_descriptor, alignment);
}

StreamDataSink::~StreamDataSink() {
  // Flush to and close the file descriptor
  if (file_descriptor_ >= 0) {
    flush();
    close(file_descriptor_);
  }
}

Result<size_t> StreamDataSink::write(const void* ptr, size_t size) {
  if (size == 0) {
    // No data to write, return current offset
    return buffer_offset_ + total_written_bytes_;
  }

  const uint8_t* data_ptr = static_cast<const uint8_t*>(ptr);

  // Align the buffer offset
  uint8_t* aligned_ptr =
      internal::align_pointer(buffer_.data() + buffer_offset_, alignment_);

  // Zero out the padding between data blobs
  size_t n_zero_pad = aligned_ptr - (buffer_.data() + buffer_offset_);
  memset(buffer_.data() + buffer_offset_, 0, n_zero_pad);

  // Calculate the new offset
  size_t new_offset = (aligned_ptr - buffer_.data()) + size;

  if (new_offset > buffer_.size()) {
    // If the new offset is out of range, flush the buffer and try again
    Result<bool> ret = flush();
    if (!ret.ok()) {
      return ret.error();
    }

    aligned_ptr = internal::align_pointer(buffer_.data(), alignment_);
    n_zero_pad = aligned_ptr - buffer_.data();
    memset(buffer_.data(), 0, n_zero_pad);
    new_offset = (aligned_ptr - buffer_.data()) + size;

    if (new_offset > buffer_.size()) {
      return Error::OutOfResources;
    }
  }

  // Copy data to the aligned position
  std::memcpy(aligned_ptr, data_ptr, size);
  buffer_offset_ = new_offset;

  return aligned_ptr - buffer_.data() + total_written_bytes_;
}

size_t StreamDataSink::get_used_bytes() const {
  return total_written_bytes_ + buffer_offset_;
}

Result<bool> StreamDataSink::flush() {
  if (buffer_offset_ > 0) {
    ssize_t written = ::write(file_descriptor_, buffer_.data(), buffer_offset_);
    if (written != buffer_offset_) {
      return Error::Internal;
    }

    total_written_bytes_ += written;
    buffer_offset_ = 0;
  }

  return true;
}

} // namespace etdump
} // namespace executorch
