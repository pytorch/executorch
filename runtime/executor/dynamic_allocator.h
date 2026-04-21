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
 * Interface for dynamic memory allocation used by DYNAMIC_UNBOUND tensors.
 *
 * Tensors marked as DYNAMIC_UNBOUND have their memory allocated lazily at
 * runtime rather than at load time. This interface allows plugging in custom
 * allocation strategies (e.g., page-aligned, tracking, virtual memory).
 */
class DynamicAllocator {
 public:
  virtual ~DynamicAllocator() = default;

  /**
   * Allocate memory.
   *
   * @param[in] size Minimum number of bytes to allocate.
   * @param[in] alignment Required alignment of the returned pointer.
   * @param[out] actual_size If non-null, receives the actual allocation size
   *     (may be larger than requested, e.g., due to growth policy).
   * @returns Pointer to allocated memory, or nullptr on failure.
   */
  virtual void* allocate(
      size_t size,
      size_t alignment,
      size_t* actual_size) = 0;

  /**
   * Reallocate memory, potentially growing the buffer.
   *
   * The allocator may implement a growth policy (e.g., 2x) so that the
   * actual allocation exceeds new_size. Old data up to min(old_size, new_size)
   * is preserved.
   *
   * @param[in] ptr Pointer previously returned by allocate() or reallocate(),
   *     or nullptr (in which case this behaves like allocate()).
   * @param[in] old_size Size of the existing allocation at ptr.
   * @param[in] new_size Minimum number of bytes needed.
   * @param[in] alignment Required alignment of the returned pointer.
   * @param[out] actual_size If non-null, receives the actual allocation size.
   * @returns Pointer to reallocated memory, or nullptr on failure. On failure,
   *     the old allocation at ptr remains valid.
   */
  virtual void* reallocate(
      void* ptr,
      size_t old_size,
      size_t new_size,
      size_t alignment,
      size_t* actual_size) = 0;

  /**
   * Free memory previously returned by allocate() or reallocate().
   *
   * @param[in] ptr Pointer to free. May be nullptr (no-op).
   */
  virtual void free(void* ptr) = 0;

  /**
   * Returns the usable capacity of a live allocation.
   *
   * @param[in] ptr Pointer previously returned by allocate() or reallocate(),
   *     or nullptr (returns 0).
   * @returns The number of usable bytes at ptr.
   */
  virtual size_t allocated_size(void* ptr) const = 0;
};

} // namespace runtime
} // namespace executorch
