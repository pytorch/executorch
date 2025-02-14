// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <cstdlib>

namespace executorch {
namespace etdump {

/**
 * DataSinkBase is an abstract class that users can inherit and implement
 * to customize the storage and management of debug data in ETDumpGen. This
 * class provides an basic and essential interface for writing tensor data to a
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
   * Write tensor data into the debug storage. This method should be implemented
   * by derived classes to handle the specifics of data storage.
   *
   * @param[in] tensor The tensor data to be written into the storage.
   * @return The offset of the starting location of the tensor data within the
   *         debug storage, which will be recorded in corresponding tensor
   * metadata of ETDump.
   */
  virtual size_t write_tensor(const executorch::aten::Tensor& tensor) = 0;
  /**
   * Get the maximum capacity of the debug storage in bytes.
   *
   * @return The total size of the debug storage.
   */
  virtual size_t get_storage_size() const = 0;
  /**
   * Get the number of bytes currently used in the debug storage.
   *
   * @return The amount of data currently stored in bytes.
   */
  virtual size_t get_used_bytes() const = 0;
};

} // namespace etdump
} // namespace executorch
