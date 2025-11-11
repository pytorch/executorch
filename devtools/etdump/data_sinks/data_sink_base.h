/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace etdump {

/**
 * DataSinkBase is an abstract class that users can inherit and implement
 * to customize the storage and management of debug data in ETDumpGen. This
 * class provides a basic and essential interface for writing datablob to a
 * user-defined storage, retrieving storage capacity, and tracking the amount of
 * data stored.
 */
class DataSinkBase {
 public:
  /**
   * Virtual destructor to ensure proper cleanup of derived classes.
   */

  virtual ~DataSinkBase() = default;
  /**
   * Write data into the debug storage. This method should be implemented
   * by derived classes to handle the specifics of data storage.
   *
   * This function should return the offset of the starting location of the
   * data within the debug storage if the write operation succeeds, or an
   * Error code if any issue occurs during the write process.
   *
   * @param[in] ptr A pointer to the data to be written into the storage.
   * @param[in] length The size of the data in bytes.
   * @return A Result object containing either:
   *         - The offset of the starting location of the data within the
   *           debug storage, which will be recorded in the corresponding
   *           metadata of ETDump, or
   *         - An error code indicating the failure reason, if any issue
   *           occurs during the write process.
   */
  virtual ::executorch::runtime::Result<size_t> write(
      const void* ptr,
      size_t length) = 0;

  /**
   * Get the number of bytes currently used in the debug storage.
   *
   * @return The amount of data currently stored in bytes.
   */
  virtual size_t get_used_bytes() const = 0;
};

} // namespace etdump
} // namespace executorch
