/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>

namespace executorch {
namespace extension {

/**
 * RAII helper that frees a set of buffers when destroyed. Movable.
 */
class BufferCleanup final {
 public:
  /**
   * Takes ownership of `buffers.data()` and the elements of `buffers`, which
   * each will be passed to `free()` when the object is destroyed.
   */
  explicit BufferCleanup(executorch::runtime::Span<void*> buffers)
      : buffers_(buffers) {}

  /**
   * Move ctor. Takes ownership of the data previously owned by `rhs`, leaving
   * `rhs` with an empty list of buffers.
   */
  BufferCleanup(BufferCleanup&& rhs) noexcept : buffers_(rhs.buffers_) {
    rhs.buffers_ = executorch::runtime::Span<void*>();
  }

  ~BufferCleanup() {
    for (auto buffer : buffers_) {
      free(buffer);
    }
    free(buffers_.data());
  }

 private:
  // Delete other rule-of-five methods.
  BufferCleanup(const BufferCleanup&) = delete;
  BufferCleanup& operator=(const BufferCleanup&) = delete;
  BufferCleanup& operator=(BufferCleanup&&) noexcept = delete;

  executorch::runtime::Span<void*> buffers_;
};

/**
 * Allocates input tensors for the provided Method, filling them with ones. Does
 * not modify inputs that are not Tensors.
 *
 * @param[in] method The Method that owns the inputs to prepare.
 *
 * @returns On success, an object that owns any allocated tensor memory. It must
 *     remain alive when calling `method->execute()`.
 * @returns An error on failure.
 */
executorch::runtime::Result<BufferCleanup> prepare_input_tensors(
    executorch::runtime::Method& method);

namespace internal {
/**
 * INTERNAL-ONLY: Creates a Tensor using the provided shape and buffer,
 * fills it with ones, and sets the input at `input_index`.
 */
executorch::runtime::Error fill_and_set_input(
    executorch::runtime::Method& method,
    executorch::runtime::TensorInfo& tensor_meta,
    size_t input_index,
    void* data_ptr);
} // namespace internal

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::BufferCleanup;
using ::executorch::extension::prepare_input_tensors;
} // namespace util
} // namespace executor
} // namespace torch
