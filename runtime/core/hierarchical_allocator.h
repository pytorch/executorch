/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/log.h>
#include <cstdint>

namespace executorch {
namespace runtime {

/**
 * A group of buffers that can be used to represent a device's memory hierarchy.
 */
class HierarchicalAllocator final {
 public:
  /**
   * Constructs a new hierarchical allocator with the given array of buffers.
   *
   * - Memory IDs are based on the index into `buffers`: `buffers[N]` will have
   *   a memory ID of `N`.
   * - `buffers.size()` must be >= `MethodMeta::num_non_const_buffers()`.
   * - `buffers[N].size()` must be >= `MethodMeta::non_const_buffer_size(N)`.
   */
  explicit HierarchicalAllocator(Span<Span<uint8_t>> buffers)
      : buffers_(buffers) {}

  /**
   * DEPRECATED: Use spans instead.
   */
  __ET_DEPRECATED HierarchicalAllocator(
      uint32_t n_allocators,
      MemoryAllocator* allocators)
      : buffers_(to_spans(n_allocators, allocators)) {}

  /**
   * Returns the address at the byte offset `offset_bytes` from the given
   * buffer's base address, which points to at least `size_bytes` of memory.
   *
   * @param[in] memory_id The ID of the buffer in the hierarchy.
   * @param[in] offset_bytes The offset in bytes into the specified buffer.
   * @param[in] size_bytes The amount of memory that should be available at
   *     the offset.
   *
   * @returns On success, the address of the requested byte offset into the
   *     specified buffer. On failure, a non-Ok Error.
   */
  __ET_NODISCARD Result<void*> get_offset_address(
      uint32_t memory_id,
      size_t offset_bytes,
      size_t size_bytes) {
    ET_CHECK_OR_RETURN_ERROR(
        memory_id < buffers_.size(),
        InvalidArgument,
        "id %" PRIu32 " >= %zu",
        memory_id,
        buffers_.size());
    Span<uint8_t> buffer = buffers_[memory_id];
    ET_CHECK_OR_RETURN_ERROR(
        offset_bytes + size_bytes <= buffer.size(),
        MemoryAllocationFailed,
        "offset_bytes (%zu) + size_bytes (%zu) >= allocator size (%zu) "
        "for memory_id %" PRIu32,
        offset_bytes,
        size_bytes,
        buffer.size(),
        memory_id);
    return buffer.data() + offset_bytes;
  }

 private:
  // TODO(T162089316): Remove the span array and to_spans once all users move to
  // spans. This array is necessary to hold the pointers and sizes that were
  // originally provided as MemoryAllocator instances.
  static constexpr size_t kSpanArraySize = 16;
  // NOTE: span_array_ must be declared before buffers_ so that it isn't
  // re-initialized to zeros after initializing buffers_.
  Span<uint8_t> span_array_[kSpanArraySize];
  Span<Span<uint8_t>> to_spans(
      uint32_t n_allocators,
      MemoryAllocator* allocators) {
    ET_CHECK_MSG(
        n_allocators <= kSpanArraySize,
        "n_allocators %" PRIu32 " > %zu",
        n_allocators,
        kSpanArraySize);
    for (uint32_t i = 0; i < n_allocators; ++i) {
      span_array_[i] =
          Span<uint8_t>(allocators[i].base_address(), allocators[i].size());
    }
    return {span_array_, n_allocators};
  }

  /// The underlying buffers.
  Span<Span<uint8_t>> buffers_;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::HierarchicalAllocator;
} // namespace executor
} // namespace torch
