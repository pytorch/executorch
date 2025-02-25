/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/devtools/etdump/data_sinks/data_sink_base.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/span.h>
#include <unistd.h> // For file operations

namespace executorch {
namespace etdump {
/**
 * StreamDataSink is a concrete implementation of DataSinkBase that manages
 * the storage of data blobs using a buffer and a file descriptor. It writes
 * data to a buffer first and flushes it to a file when the buffer is full,
 * or when the flush() method is called explicitly. It is useful for storing
 * large amounts of data in a file without consuming excessive memory.
 *
 * Noted that
 * - This class is demonstrated purpose only, and not intended to be used or
 * existed in production.
 * - The buffer is provided and owned by user at construction time. The user is
 * responsible for ensuring the validity and ownership of these resources.
 */
class StreamDataSink : public DataSinkBase {
 public:
  /**
   * Creates a StreamDataSink with a given buffer and file path.
   *
   * @param[in] buffer_data_ptr A pointer to the buffer used for temporary
   * storage.
   * @param[in] buffer_size The size of the buffer.
   * @param[in] file_path The path to the file for writing data.
   * @param[in] alignment The alignment requirement for the buffer. Default
   * is 64.
   * @returns A new StreamDataSink on success.
   * @retval Error::InvalidArgument `alignment` is not a power of two.
   * @retval Error::AccessFailed Cannot access/create the file using
   * `file_path`.
   */
  static ::executorch::runtime::Result<StreamDataSink> create(
      void* buffer_data_ptr,
      size_t buffer_size,
      const char* file_path,
      size_t alignment = 64);

  /**
   * Destructor that ensures all buffered data is flushed to the file.
   */
  ~StreamDataSink() override;

  // Delete copy constructor and copy assignment operator
  StreamDataSink(const StreamDataSink&) = delete;
  StreamDataSink& operator=(const StreamDataSink&) = delete;

  StreamDataSink(StreamDataSink&& other) noexcept;
  StreamDataSink& operator=(StreamDataSink&& other) = default;

  /**
   * Writes data into the debug storage aligned to the given alignment.
   *
   * @param[in] ptr A pointer to the data to be written into the storage.
   * @param[in] size The size of the data in bytes.
   * @return A Result object containing either:
   *         - The offset of the starting location of the data within the
   *           debug storage, or
   *         - An error code indicating the failure reason.
   */
  ::executorch::runtime::Result<size_t> write(const void* ptr, size_t size)
      override;

  /**
   * Gets the number of bytes currently used in the debug storage.
   *
   * @return The amount of data currently stored in bytes.
   */
  size_t get_used_bytes() const override;

  /**
   * Flushes all buffered data to the file. No alignment is applied.
   *
   * @return A Result object containing either:
   *         - true, indicating the flush operation was successful, or
   *         - An error code indicating the failure reason.
   */
  ::executorch::runtime::Result<bool> flush();

 private:
  /**
   * Constructs a StreamDataSink with a given buffer and file descriptor.
   *
   * @param[in] buffer A span representing the buffer used for temporary
   * storage.
   * @param[in] file_descriptor A valid file descriptor for writing data.
   */
  StreamDataSink(
      void* buffer_data_ptr,
      size_t buffer_size,
      int file_descriptor,
      size_t alignment)
      : buffer_({static_cast<uint8_t*>(buffer_data_ptr), buffer_size}),
        buffer_offset_(0),
        file_descriptor_(file_descriptor),
        total_written_bytes_(0),
        alignment_(alignment) {}
  ::executorch::runtime::Span<uint8_t> buffer_;
  size_t buffer_offset_;
  int file_descriptor_;
  size_t total_written_bytes_;
  size_t alignment_;
};
} // namespace etdump
} // namespace executorch
