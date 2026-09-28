/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <executorch/extension/backend_data/data_writer.h>

namespace executorch {
namespace extension {

class FileDataWriterState;

/**
 * An out-of-place DataWriter that atomically replaces a file at publish().
 *
 * The first write lazily creates a temporary file. Each write uses positional
 * I/O at the requested absolute offset and returns only after copying the
 * caller's bytes. The destination remains unchanged until publish().
 *
 * The temporary output is created in `temporary_directory`, or in the
 * destination's directory when that string is empty. Atomic publication
 * requires the temporary output and destination to be on the same filesystem.
 * publish() truncates the output to its logical extent, synchronizes and closes
 * it, atomically renames it over the destination, and synchronizes the parent
 * directory on POSIX. Destruction removes an unpublished temporary file.
 */
class FileDataWriter final : public DataWriter {
 public:
  /**
   * Creates a writer for `file_name` without opening or modifying it.
   *
   * @param[in] file_name Destination path that publish() will replace.
   * @param[in] temporary_directory Directory for temporary output. An empty
   *     value selects the destination's directory.
   */
  explicit FileDataWriter(
      std::string file_name,
      std::string temporary_directory = std::string());

  /** Discards any unpublished temporary output. */
  ~FileDataWriter() override;

  /** Lazily creates the temporary file and synchronously writes `data`. */
  ET_NODISCARD executorch::runtime::Error write(
      executorch::runtime::Span<const uint8_t> data,
      size_t offset) override;

  /** Synchronizes, closes, and atomically publishes the temporary file. */
  ET_NODISCARD executorch::runtime::Error publish() override;

 private:
  std::string file_name_;
  std::string temporary_directory_;
  std::unique_ptr<FileDataWriterState> state_;
  bool published_{false};
};

} // namespace extension
} // namespace executorch
