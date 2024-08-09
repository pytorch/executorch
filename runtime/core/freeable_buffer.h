/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

namespace executorch {
namespace runtime {

/**
 * A read-only buffer than can be freed.
 */
class FreeableBuffer final {
 public:
  // Callback signature for the function that does the freeing.
  using FreeFn = void (*)(void* context, void* data, size_t size);

  /**
   * Creates an empty FreeableBuffer with size zero and a null data pointer.
   */
  FreeableBuffer()
      : free_fn_(nullptr),
        free_fn_context_(nullptr),
        data_(nullptr),
        size_(0) {}

  /**
   * Creates a FreeableBuffer with an optional free function.
   *
   * @param[in] data The data of the segment.
   * @param[in] size The size of the segment data, in bytes.
   * @param[in] free_fn Optional function to free the data. Guaranteed to be
   *     called exactly once before the FreeableBuffer is destroyed. May be
   *     nullptr. NOTE: This function must be thread-safe. If it modifies common
   *     state, the function must do its own locking.
   * @param[in] free_fn_context Opaque pointer to pass as the `context`
   *     parameter of `free_fn`. May be nullptr.
   */
  FreeableBuffer(
      const void* data,
      size_t size,
      FreeFn free_fn,
      void* free_fn_context = nullptr)
      : free_fn_(free_fn),
        free_fn_context_(free_fn_context),
        data_(data),
        size_(size) {}

  /**
   * Move ctor. Takes the ownership of the data previously owned by `rhs`,
   * leaving `rhs` pointing to nullptr.
   */
  FreeableBuffer(FreeableBuffer&& rhs) noexcept
      : free_fn_(rhs.free_fn_),
        free_fn_context_(rhs.free_fn_context_),
        data_(rhs.data_),
        size_(rhs.size_) {
    rhs.free_fn_ = nullptr;
    rhs.free_fn_context_ = nullptr;
    rhs.data_ = nullptr;
    rhs.size_ = 0;
  }

  ~FreeableBuffer() {
    Free();
  }

  /**
   * Frees the data if not already free. Safe to call multiple times.
   */
  void Free() {
    if (data_ != nullptr) {
      if (free_fn_ != nullptr) {
        free_fn_(free_fn_context_, const_cast<void*>(data_), size_);
      }
      data_ = nullptr;
      size_ = 0;
    }
  }

  /**
   * Size of the data in bytes. Returns 0 if the data has been freed.
   */
  size_t size() const {
    return size_;
  }

  /**
   * Pointer to the data. Returns nullptr if the data has been freed.
   */
  const void* data() const {
    return data_;
  }

 private:
  // Delete other rule-of-five methods.
  FreeableBuffer(const FreeableBuffer& rhs) = delete;
  FreeableBuffer& operator=(FreeableBuffer&& rhs) noexcept = delete;
  FreeableBuffer& operator=(const FreeableBuffer& rhs) = delete;

  FreeFn free_fn_;
  void* free_fn_context_;
  const void* data_;
  size_t size_;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::FreeableBuffer;
} // namespace executor
} // namespace torch
