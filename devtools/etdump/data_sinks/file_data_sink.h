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
#include <cstdio> // For FILE operations

namespace executorch {
namespace etdump {

/**
 * FileDataSink is a concrete implementation of the DataSinkBase class,
 * designed to facilitate the direct writing of data to a file. It is
 * particularly useful for scenarios where immediate data storage is
 * required, such as logging or streaming data to a file for real-time
 * analysis. The class manages file operations, including opening, writing,
 * and closing the file, while handling potential errors during these
 * operations.
 */

class FileDataSink : public DataSinkBase {
 public:
  /**
   * Creates a FileDataSink with a given file path.
   *
   * @param[in] file_path The path to the file for writing data.
   * @return A Result object containing either:
   *         - A FileDataSink object if succees, or
   *         - AccessFailed Error when the file cannot be accessed or created
   */
  static ::executorch::runtime::Result<FileDataSink> create(
      const char* file_path);

  /**
   * Destructor that closes the file.
   */
  ~FileDataSink() override;

  // Delete copy constructor and copy assignment operator
  FileDataSink(const FileDataSink&) = delete;
  FileDataSink& operator=(const FileDataSink&) = delete;

  FileDataSink(FileDataSink&& other) noexcept;
  FileDataSink& operator=(FileDataSink&& other) = default;

  /**
   * Writes data directly to the file.
   *
   * This function does not perform any alignment, and will overwrite
   * any existing data in the file.
   *
   * @param[in] ptr A pointer to the data to be written into the file.
   * @param[in] size The size of the data in bytes.
   * @return A Result object containing either:
   *         - The offset of the starting location of the data within the
   *           file, or
   *         - AccessFailedError if the file has been closed.
   *         - InternalError if the os write operation fails.
   */
  ::executorch::runtime::Result<size_t> write(const void* ptr, size_t size)
      override;

  /**
   * Gets the number of bytes currently written to the file.
   *
   * @return The amount of data currently stored in bytes.
   */
  size_t get_used_bytes() const override;

  /**
   * Closes the file, if it is open.
   */
  void close();

 private:
  /**
   * Constructs a FileDataSink with a given file pointer.
   *
   * @param[in] file A valid file pointer for writing data.
   */
  explicit FileDataSink(FILE* file) : file_(file), total_written_bytes_(0) {}

  FILE* file_;
  size_t total_written_bytes_;
};

} // namespace etdump
} // namespace executorch
