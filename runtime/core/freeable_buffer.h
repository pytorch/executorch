/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {

/**
 * A read-only buffer than can be freed.
 */
class FreeableBuffer final {
 public:
  // Callback signature for the function that does the freeing.
  using FreeFn = void (*)(void* context, void* data, size_t size);
  using FreeUInt64Fn =
      void (*)(void* context, uint64_t data_uint64, size_t size);

 private:
  // Forward declare types.
  struct PointerData {
    const void* data_;
    FreeFn free_fn_;
  };

  struct UInt64Data {
    // A pointer value cast to uint64_t.
    uint64_t data_;
    FreeUInt64Fn free_fn_;
  };

 public:
  /**
   * Creates an empty FreeableBuffer with size zero and a null data pointer.
   */
  FreeableBuffer()
      : data_(PointerData{nullptr, nullptr}),
        free_fn_context_(nullptr),
        size_(0) {}

  /**
   * Creates a FreeableBuffer with an optional free function.
   *
   * @param[in] data The data of the segment, as a void*.
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
      : data_(PointerData{data, free_fn}),
        free_fn_context_(free_fn_context),
        size_(size) {}

  /**
   * Creates a FreeableBuffer with an optional free function.
   *
   * NOTE: most users should use the other ctor with FreeFn.
   * This variant exists for situations where the FreeableBuffer points to
   * memory on a different core whose pointer value is larger than the local
   * core's void*.
   *
   * @param[in] data Pointer to the data of the segment, cast to a uint64_t
   * value.
   * @param[in] size The size of the segment data, in bytes.
   * @param[in] free_fn Optional function to free the data. Guaranteed to be
   *     called exactly once before the FreeableBuffer is destroyed. May be
   *     nullptr. NOTE: This function must be thread-safe. If it modifies common
   *     state, the function must do its own locking.
   * @param[in] free_fn_context Opaque pointer to pass as the `context`
   *     parameter of `free_fn`. May be nullptr.
   */
  explicit FreeableBuffer(
      const uint64_t data_uint64,
      size_t size,
      FreeUInt64Fn free_fn,
      void* free_fn_context = nullptr)
      : data_(UInt64Data{data_uint64, free_fn}),
        free_fn_context_(free_fn_context),
        size_(size) {}

  /**
   * Move ctor. Takes the ownership of the data previously owned by `rhs`,
   * leaving `rhs` pointing to nullptr.
   */
  FreeableBuffer(FreeableBuffer&& rhs) noexcept
      : data_(rhs.data_),
        free_fn_context_(rhs.free_fn_context_),
        size_(rhs.size_) {
    if (std::holds_alternative<PointerData>(rhs.data_)) {
      rhs.data_ = PointerData{nullptr, nullptr};
    } else {
      rhs.data_ = UInt64Data{0, nullptr};
    }
    rhs.free_fn_context_ = nullptr;
    rhs.size_ = 0;
  }

  ~FreeableBuffer() {
    Free();
  }

  /**
   * Frees the data if not already free. Safe to call multiple times.
   */
  void Free() {
    if (std::holds_alternative<PointerData>(data_)) {
      PointerData& ptr_data = std::get<PointerData>(data_);
      if (ptr_data.data_ != nullptr && ptr_data.free_fn_ != nullptr) {
        // Do not need to check for truncation here, as free_fn_ is only set
        // using the void* ctor.
        ptr_data.free_fn_(
            free_fn_context_, const_cast<void*>(ptr_data.data_), size_);
      }
      ptr_data.data_ = nullptr;
      size_ = 0;
    } else {
      UInt64Data& int64_data = std::get<UInt64Data>(data_);
      if (int64_data.data_ != 0 && int64_data.free_fn_ != nullptr) {
        int64_data.free_fn_(free_fn_context_, int64_data.data_, size_);
      }
      int64_data.data_ = static_cast<uint64_t>(0);
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
    ET_CHECK_MSG(
        std::holds_alternative<PointerData>(data_),
        "FreeableBuffer is backed by an uint64_t, please use the data_uint64_type() API.");
    return std::get<PointerData>(data_).data_;
  }

  /**
   * Pointer to the data. Returns nullptr if the data has been freed.
   * Safe version of data() API that returns an ERror if the data is
   * backed by int64_t instead of void*.
   */
  Result<const void*> data_safe() const {
    ET_CHECK_OR_RETURN_ERROR(
        std::holds_alternative<PointerData>(data_),
        InvalidType,
        "FreeableBuffer is backed by an uint64_t, please use the data_uint64_type() API.");
    return std::get<PointerData>(data_).data_;
  }

  /**
   * Data address as a uint64_t. Returns zero if the data has been freed.
   * Most users should use data(). data_uint64_type() is only helpful in
   * situations where the FreeableBuffer points to memory on a different core
   * whose pointer value is larger than the local core's void *.
   */
  Result<uint64_t> data_uint64_type() const {
    ET_CHECK_OR_RETURN_ERROR(
        std::holds_alternative<UInt64Data>(data_),
        InvalidType,
        "FreeableBuffer is backed by a void*, please use the data() API.");
    return std::get<UInt64Data>(data_).data_;
  }

 private:
  // Delete other rule-of-five methods.
  FreeableBuffer(const FreeableBuffer& rhs) = delete;
  FreeableBuffer& operator=(FreeableBuffer&& rhs) noexcept = delete;
  FreeableBuffer& operator=(const FreeableBuffer& rhs) = delete;

  // This stores either a PointerData or a UInt64Data structure. Most users
  // should use the PointerData variant and the void* ctor. This creates a
  // FreeableBuffer backed by void*, accessed using the void* getter data().
  // The UInt64Data variant is only helpful in situations where the
  // FreeableBuffer points to memory on a different core whose pointer value
  // is larger than the local core's void*.
  std::variant<PointerData, UInt64Data> data_;

  void* free_fn_context_;
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
