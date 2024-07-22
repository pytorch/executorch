/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace torch {
namespace executor {

/**
 * Loads from a data source.
 *
 * See //executorch/extension/data_loader for common implementations.
 */
class DataLoader {
 public:
  /**
   * Describes the content of the segment.
   */
  struct SegmentInfo {
    /**
     * Represents the purpose of the segment.
     */
    enum class Type {
      /**
       * Data for the actual program.
       */
      Program,
      /**
       * Holds constant tensor data.
       */
      Constant,
      /**
       * Data used for initializing a backend.
       */
      Backend,
    };

    /// Type of the segment.
    Type segment_type;

    /// Index of the segment within the segment list. Undefined for program
    /// segments.
    size_t segment_index;

    /// An optional, null-terminated string describing the segment. For
    /// `Backend` segments, this is the backend ID. Null for other segment
    /// types.
    const char* descriptor;

    SegmentInfo() = default;

    explicit SegmentInfo(
        Type segment_type,
        size_t segment_index = 0,
        const char* descriptor = nullptr)
        : segment_type(segment_type),
          segment_index(segment_index),
          descriptor(descriptor) {}
  };

  virtual ~DataLoader() = default;

  /**
   * Loads data from the underlying data source.
   *
   * NOTE: This must be thread-safe. If this call modifies common state, the
   * implementation must do its own locking.
   *
   * @param offset The byte offset in the data source to start loading from.
   * @param size The number of bytes to load.
   * @param segment_info Information about the segment being loaded.
   *
   * @returns a `FreeableBuffer` that owns the loaded data.
   */
  __ET_NODISCARD virtual Result<FreeableBuffer>
  load(size_t offset, size_t size, const SegmentInfo& segment_info) = 0;

  /**
   * Returns the length of the underlying data source, typically the file size.
   */
  __ET_NODISCARD virtual Result<size_t> size() const = 0;
};

} // namespace executor
} // namespace torch
