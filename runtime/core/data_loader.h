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
 */
class DataLoader {
 public:
  virtual ~DataLoader() = default;

  /**
   * Loads `size` bytes at byte offset `offset` from the underlying data source
   * into a `FreeableBuffer`, which owns the memory.
   *
   * NOTE: This must be thread-safe. If this call modifies common state, the
   * implementation must do its own locking.
   */
  __ET_NODISCARD virtual Result<FreeableBuffer> Load(
      size_t offset,
      size_t size) = 0;

  /**
   * Returns the length of the underlying data source, typically the file size.
   */
  __ET_NODISCARD virtual Result<size_t> size() const = 0;
};

} // namespace executor
} // namespace torch
