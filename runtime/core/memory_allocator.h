/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdio.h>
#include <cinttypes>
#include <cstdint>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/log.h>
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
        end_(base_address + size),
        cur_(base_address),
        size_(size) {}

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
    if (!isPowerOf2(alignment)) {
      ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
      return nullptr;
    }

    // The allocation will occupy [start, end), where the start is the next
    // position that's a multiple of alignment.
    uint8_t* start = alignPointer(cur_, alignment);
    uint8_t* end = start + size;

    // If the end of this allocation exceeds the end of this allocator, print
    // error messages and return nullptr
    if (end > end_) {
      ET_LOG(
          Error,
          "Memory allocation failed: %zuB requested (adjusted for alignment), %zuB available",
          static_cast<size_t>(end - cur_),
          static_cast<size_t>(end_ - cur_));
      return nullptr;
    }

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
    return static_cast<T*>(this->allocate(size * sizeof(T), alignment));
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

  void enable_profiling(__ET_UNUSED const char* name) {
    prof_id_ = EXECUTORCH_TRACK_ALLOCATOR(name);
  }

  virtual ~MemoryAllocator() {}

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
  static bool isPowerOf2(size_t value) {
    return value > 0 && (value & ~(value - 1)) == value;
  }

  /**
   * Returns the next alignment for a given pointer.
   */
  static uint8_t* alignPointer(void* ptr, size_t alignment) {
    intptr_t addr = reinterpret_cast<intptr_t>(ptr);
    if ((addr & (alignment - 1)) == 0) {
      // Already aligned.
      return reinterpret_cast<uint8_t*>(ptr);
    }
    addr = (addr | (alignment - 1)) + 1;
    return reinterpret_cast<uint8_t*>(addr);
  }

 private:
  uint8_t* const begin_;
  uint8_t* const end_;
  uint8_t* cur_;
  uint32_t const size_;
  int32_t prof_id_ = -1;
};

#if __ET_HAVE_GNU_STATEMENT_EXPRESSIONS
/**
 * Tries allocating from the specified MemoryAllocator*.
 *
 * - On success, returns a pointer to the allocated buffer.
 * - On failure, executes the provided code block, which must return or panic.
 *
 * Example:
 * @code
 *   char* buf = ET_TRY_ALLOCATE_OR(
 *       memory_allocator, bufsize, {
 *         *out_err = Error::MemoryAllocationFailed;
 *         return nullopt;
 *       });
 * @endcode
 */
#define ET_TRY_ALLOCATE_OR(memory_allocator__, nbytes__, ...)              \
  ({                                                                       \
    void* et_try_allocate_result = memory_allocator__->allocate(nbytes__); \
    if (et_try_allocate_result == nullptr && nbytes__ > 0) {               \
      __VA_ARGS__                                                          \
      /* The args must return. */                                          \
      __ET_UNREACHABLE();                                                  \
    }                                                                      \
    et_try_allocate_result;                                                \
  })

/**
 * Tries allocating an instance of type__ from the specified MemoryAllocator*.
 *
 * - On success, returns a pointer to the allocated buffer. Note that the memory
 *   will not be initialized.
 * - On failure, executes the provided code block, which must return or panic.
 *
 * Example:
 * @code
 *   char* buf = ET_TRY_ALLOCATE_INSTANCE_OR(
 *       memory_allocator,
 *       MyType,
 *       { *out_err = Error::MemoryAllocationFailed; return nullopt; });
 * @endcode
 */
#define ET_TRY_ALLOCATE_INSTANCE_OR(memory_allocator__, type__, ...) \
  ({                                                                 \
    type__* et_try_allocate_result =                                 \
        memory_allocator__->allocateInstance<type__>();              \
    if (et_try_allocate_result == nullptr) {                         \
      __VA_ARGS__                                                    \
      /* The args must return. */                                    \
      __ET_UNREACHABLE();                                            \
    }                                                                \
    et_try_allocate_result;                                          \
  })

/**
 * Tries allocating multiple elements of a given type from the specified
 * MemoryAllocator*.
 *
 * - On success, returns a pointer to the allocated buffer.
 * - On failure, executes the provided code block, which must return or panic.
 *
 * Example:
 * @code
 *   Tensor* tensor_list = ET_TRY_ALLOCATE_LIST_OR(
 *       memory_allocator, Tensor, num_tensors, {
 *         *out_err = Error::MemoryAllocationFailed;
 *         return nullopt;
 *       });
 * @endcode
 */
#define ET_TRY_ALLOCATE_LIST_OR(memory_allocator__, type__, nelem__, ...) \
  ({                                                                      \
    type__* et_try_allocate_result =                                      \
        memory_allocator__->allocateList<type__>(nelem__);                \
    if (et_try_allocate_result == nullptr && nelem__ > 0) {               \
      __VA_ARGS__                                                         \
      /* The args must return. */                                         \
      __ET_UNREACHABLE();                                                 \
    }                                                                     \
    et_try_allocate_result;                                               \
  })
#else // !__ET_HAVE_GNU_STATEMENT_EXPRESSIONS
/**
 * The recommended alternative for statement expression-incompatible compilers
 * is to directly allocate the memory.
 * e.g. memory_allocator__->allocate(nbytes__);
 */
#define ET_TRY_ALLOCATE_OR(memory_allocator__, nbytes__, ...) \
  static_assert(                                              \
      false,                                                  \
      "ET_TRY_ALLOCATE_OR uses statement expressions and \
      thus is not available for use with this compiler.");

/**
 * The recommended alternative for statement expression-incompatible compilers
 * is to directly allocate the memory.
 * e.g. memory_allocator__->allocateInstance<type__>();
 */
#define ET_TRY_ALLOCATE_INSTANCE_OR(memory_allocator__, type__, ...) \
  static_assert(                                                     \
      false,                                                         \
      "ET_TRY_ALLOCATE_INSTANCE_OR uses statement \
    expressions and thus is not available for use with this compiler.");

/**
 * The recommended alternative for statement expression-incompatible compilers
 * is to directly use allocate the memory.
 * e.g. memory_allocator__->allocateList<type__>(nelem__);
 */
#define ET_TRY_ALLOCATE_LIST_OR(memory_allocator__, type__, nelem__, ...) \
  static_assert(                                                          \
      false,                                                              \
      "ET_TRY_ALLOCATE_LIST_OR uses statement \
    expressions and thus is not available for use with this compiler.");
#endif // !__ET_HAVE_GNU_STATEMENT_EXPRESSIONS

/**
 * Tries allocating from the specified MemoryAllocator*.
 *
 * - On success, returns a pointer to the allocated buffer.
 * - On failure, returns `Error::MemoryAllocationFailed` from the calling
 *   function, which must be declared to return `executorch::runtime::Error`.
 *
 * Example:
 * @code
 *   char* buf = ET_ALLOCATE_OR_RETURN_ERROR(memory_allocator, bufsize);
 * @endcode
 */
#define ET_ALLOCATE_OR_RETURN_ERROR(memory_allocator__, nbytes__) \
  ET_TRY_ALLOCATE_OR(memory_allocator__, nbytes__, {              \
    return ::executorch::runtime::Error::MemoryAllocationFailed;  \
  })

/**
 * Tries allocating an instance of type__ from the specified MemoryAllocator*.
 *
 * - On success, returns a pointer to the allocated buffer. Note that the memory
 *   will not be initialized.
 * - On failure, returns `Error::MemoryAllocationFailed` from the calling
 *   function, which must be declared to return `executorch::runtime::Error`.
 *
 * Example:
 * @code
 *   char* buf = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(memory_allocator, MyType);
 * @endcode
 */
#define ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(memory_allocator__, type__) \
  ET_TRY_ALLOCATE_INSTANCE_OR(memory_allocator__, type__, {              \
    return ::executorch::runtime::Error::MemoryAllocationFailed;         \
  })

/**
 * Tries allocating multiple elements of a given type from the specified
 * MemoryAllocator*.
 *
 * - On success, returns a pointer to the allocated buffer.
 * - On failure, returns `Error::MemoryAllocationFailed` from the calling
 *   function, which must be declared to return `executorch::runtime::Error`.
 *
 * Example:
 * @code
 *   Tensor* tensor_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
 *       memory_allocator, Tensor, num_tensors);
 * @endcode
 */
#define ET_ALLOCATE_LIST_OR_RETURN_ERROR(memory_allocator__, type__, nelem__) \
  ET_TRY_ALLOCATE_LIST_OR(memory_allocator__, type__, nelem__, {              \
    return ::executorch::runtime::Error::MemoryAllocationFailed;              \
  })

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::MemoryAllocator;
} // namespace executor
} // namespace torch
