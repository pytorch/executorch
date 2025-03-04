/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/devtools/etdump/data_sinks/data_sink_base.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/span.h>

namespace executorch {
namespace etdump {

/**
 * BufferDataSink is a concrete implementation of the DataSinkBase class,
 * designed to store debug data in a pre-allocated, user-owned buffer. This
 * class provides methods to write raw data and tensor data into the buffer,
 * ensuring proper alignment and managing padding as needed.
 */
class BufferDataSink : public DataSinkBase {
 public:
  /**
   * Creates a BufferDataSink with a given span buffer.
   *
   * @param[in] buffer A Span object representing the buffer where data will be
   * stored.
   * @param[in] alignment The alignment requirement for the buffer. It must be
   * a power of two and greater than zero. Default is 64.
   * @return A Result object containing either:
   *         - A BufferDataSink object if succees, or
   *         - An error code indicating the failure reason, if any issue
   *           occurs during the creation process.
   */
  static ::executorch::runtime::Result<BufferDataSink> create(
      ::executorch::runtime::Span<uint8_t> buffer,
      size_t alignment = 64) noexcept;

  /**
   * Creates a BufferDataSink with a given span buffer.
   *
   * @param[in] ptr A pointer to the data blob where data will be stored.
   * @param[in] size The size of the data blob in bytes.
   * @param[in] alignment The alignment requirement for the buffer. It must be
   * a power of two and greater than zero. Default is 64.
   * @return A Result object containing either:
   *         - A BufferDataSink object if succees, or
   *         - An error code indicating the failure reason, if any issue
   *           occurs during the creation process.
   */
  static ::executorch::runtime::Result<BufferDataSink>
  create(void* ptr, size_t size, size_t alignment = 64) noexcept;

  /**
   * Creates a empty BufferDataSink;
   */
  BufferDataSink() = default;

  // Uncopiable and unassignable to avoid double assignment and free of the
  // internal buffer.
  BufferDataSink(const BufferDataSink&) = delete;
  BufferDataSink& operator=(const BufferDataSink&) = delete;

  // Movable to be compatible with Result.
  BufferDataSink(BufferDataSink&&) = default;
  BufferDataSink& operator=(BufferDataSink&&) = default;

  ~BufferDataSink() override = default;

  /**
   * Write data into the debug buffer and return the offset of the starting
   * location of the data within the buffer.
   *
   * @param[in] ptr A pointer to the data to be written into the storage.
   * @param[in] size The size of the data in bytes.
   * @return A Result object containing either:
   *         - The offset of the starting location of the data within the
   *           debug buffer, or
   *         - An error code indicating the failure reason, if any issue
   *           occurs during the write process.
   */
  ::executorch::runtime::Result<size_t> write(const void* ptr, size_t size)
      override;

  /**
   * Retrieves the total size of the buffer.
   *
   * @return A Result object containing the total size of the buffer in bytes.
   */
  ::executorch::runtime::Result<size_t> get_storage_size() const;

  /**
   * Retrieves the number of bytes currently used in the buffer.
   *
   * @return The amount of data currently stored in the buffer in bytes.
   */
  size_t get_used_bytes() const override;

 private:
  /**
   * Constructs a BufferDataSink with a given buffer.
   *
   * @param[in] buffer A Span object representing the buffer where data will be
   * stored.
   * @param[in] alignment The alignment requirement for the buffer. It must be
   * a power of two. Default is 64.
   */
  explicit BufferDataSink(
      ::executorch::runtime::Span<uint8_t> buffer,
      size_t alignment)
      : debug_buffer_(buffer), offset_(0), alignment_(alignment) {}

  // A Span object representing the buffer used for storing debug data.
  ::executorch::runtime::Span<uint8_t> debug_buffer_;

  // The offset of the next available location in the buffer.
  size_t offset_;

  // The alignment of the buffer.
  size_t alignment_;
};

} // namespace etdump
} // namespace executorch
