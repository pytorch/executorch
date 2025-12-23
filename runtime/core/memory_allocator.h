/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cinttypes>

#include <c10/util/safe_numerics.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/profiler.h>

namespace executorch {
namespace runtime {

/**
 * A class that does simple allocation based on a size and returns the pointer
 * to the memory address. It bookmarks a buffer with certain size. The
 * allocation is simply checking space and growing the cur_ pointer with each
 * allocation request.
 *
 * Simple example:
 *
 *   // User allocates a 100 byte long memory in the heap.
 *   uint8_t* memory_pool = malloc(100 * sizeof(uint8_t));
 *   MemoryAllocator allocator(100, memory_pool)
 *   // Pass allocator object in the Executor
 *
 *   Underneath the hood, ExecuTorch will call
 *   allocator.allocate() to keep iterating cur_ pointer
 */
class MemoryAllocator {
 public:
  /**
   * Default alignment of memory returned by this class. Ensures that pointer
   * fields of structs will be aligned. Larger types like `long double` may not
   * be, however, depending on the toolchain and architecture.
   */
  static constexpr size_t kDefaultAlignment = alignof(void*);

  /**
   * Constructs a new memory allocator of a given `size`, starting at the
   * provided `base_address`.
   *
   * @param[in] size The size in bytes of the buffer at `base_address`.
   * @param[in] base_address The buffer to allocate from. Does not take
   *     ownership of this buffer, so it must be valid for the lifetime of of
   *     the MemoryAllocator.
   */
  MemoryAllocator(uint32_t size, uint8_t* base_address)
      : begin_(base_address),
        end_(
            base_address
                ? (UINTPTR_MAX - reinterpret_cast<uintptr_t>(base_address) >=
                           size
                       ? base_address + size
                       : nullptr)
                : nullptr),
        cur_(base_address),
        size_(size) {
    ET_CHECK_MSG(
        base_address || size == 0,
        "Base address is null but size=%" PRIu32,
        size);
    ET_CHECK_MSG(
        !base_address || size == 0 ||
            (UINTPTR_MAX - reinterpret_cast<uintptr_t>(base_address) >= size),
        "Address space overflow in allocator");
  }

  virtual ~MemoryAllocator() = default;

  /**
   * Allocates `size` bytes of memory.
   *
   * @param[in] size Number of bytes to allocate.
   * @param[in] alignment Minimum alignment for the returned pointer. Must be a
   *     power of 2.
   *
   * @returns Aligned pointer to the allocated memory on success.
   * @retval nullptr Not enough memory, or `alignment` was not a power of 2.
   */
  virtual void* allocate(size_t size, size_t alignment = kDefaultAlignment) {
    if ET_UNLIKELY (!begin_ || !end_) {
      ET_LOG(Error, "allocate() on zero-capacity allocator");
      return nullptr;
    }
    if (!isPowerOf2(alignment)) {
      ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
      return nullptr;
    }

    // The allocation will occupy [start, end), where the start is the next
    // position that's a multiple of alignment.
    uint8_t* start = alignPointer(cur_, alignment);
    size_t padding = static_cast<size_t>(start - cur_);
    size_t available = static_cast<size_t>(end_ - cur_);
    if ET_UNLIKELY (padding > available || size > available - padding) {
      ET_LOG(
          Error,
          "Memory allocation failed: %zuB requested (adjusted for alignment), %zuB available",
          padding + size,
          available);
      return nullptr;
    }
    uint8_t* end = start + size;

    // Otherwise, record how many bytes were used, advance cur_ to the new end,
    // and then return start. Note that the number of bytes used is (end - cur_)
    // instead of (end - start) because start > cur_ if there is a misalignment
    EXECUTORCH_TRACK_ALLOCATION(prof_id_, end - cur_);
    cur_ = end;
    return static_cast<void*>(start);
  }

  /**
   * Allocates a buffer large enough for an instance of type T. Note that the
   * memory will not be initialized.
   *
   * Example:
   * @code
   *   auto p = memory_allocator->allocateInstance<MyType>();
   * @endcode
   *
   * @param[in] alignment Minimum alignment for the returned pointer. Must be a
   *     power of 2. Defaults to the natural alignment of T.
   *
   * @returns Aligned pointer to the allocated memory on success.
   * @retval nullptr Not enough memory, or `alignment` was not a power of 2.
   */
  template <typename T>
  T* allocateInstance(size_t alignment = alignof(T)) {
    return static_cast<T*>(this->allocate(sizeof(T), alignment));
  }

  /**
   * Allocates `size` number of chunks of type T, where each chunk is of size
   * equal to sizeof(T) bytes.
   *
   * @param[in] size Number of memory chunks to allocate.
   * @param[in] alignment Minimum alignment for the returned pointer. Must be a
   *     power of 2. Defaults to the natural alignment of T.
   *
   * @returns Aligned pointer to the allocated memory on success.
   * @retval nullptr Not enough memory, or `alignment` was not a power of 2.
   */
  template <typename T>
  T* allocateList(size_t size, size_t alignment = alignof(T)) {
    // Some users of this method allocate lists of pointers, causing the next
    // line to expand to `sizeof(type *)`, which triggers a clang-tidy warning.
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    size_t bytes_size = 0;
    bool overflow = c10::mul_overflows(size, sizeof(T), &bytes_size);
    if (overflow) {
      ET_LOG(
          Error,
          "Failed to allocate list: size(%zu) * sizeof(T)(%zu) overflowed",
          size,
          sizeof(T));
      return nullptr;
    }
    return static_cast<T*>(this->allocate(bytes_size, alignment));
  }

  // Returns the allocator memory's base address.
  virtual uint8_t* base_address() const {
    return begin_;
  }

  // Returns the total size of the allocator's memory buffer.
  virtual uint32_t size() const {
    return size_;
  }

  // Resets the current pointer to the base address. It does nothing to
  // the contents.
  virtual void reset() {
    cur_ = begin_;
  }

  void enable_profiling(ET_UNUSED const char* name) {
    prof_id_ = EXECUTORCH_TRACK_ALLOCATOR(name);
  }

 protected:
  /**
   * Returns the profiler ID for this allocator.
   */
  int32_t prof_id() const {
    return prof_id_;
  }

  /**
   * Returns true if the value is an integer power of 2.
   */
  static constexpr bool isPowerOf2(size_t value) {
    return value && !(value & (value - 1));
  }

  /**
   * Returns the next alignment for a given pointer.
   */
  static inline uint8_t* alignPointer(void* ptr, size_t alignment) {
    uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t mask = static_cast<uintptr_t>(alignment - 1);
    address = (address + mask) & ~mask;
    return reinterpret_cast<uint8_t*>(address);
  }

 private:
  uint8_t* const begin_;
  uint8_t* const end_;
  uint8_t* cur_;
  uint32_t const size_;
  int32_t prof_id_ = -1;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::MemoryAllocator;
} // namespace executor
} // namespace torch
