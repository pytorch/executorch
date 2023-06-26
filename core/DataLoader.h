// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

#include <executorch/compiler/Compiler.h>
#include <executorch/core/FreeableBuffer.h>
#include <executorch/core/Result.h>

namespace torch {
namespace executor {

/**
 * Loads from a data source.
 *
 * See //executorch/util for common implementations.
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
