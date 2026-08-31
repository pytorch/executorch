/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/irange.h>
#include <c10/util/safe_numerics.h>

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/portable_type/device.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

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
   * Constructs a new hierarchical allocator with per-buffer device metadata.
   *
   * @param[in] buffers Same as above. May contain a mix of CPU and device
   *     pointers — HierarchicalAllocator only does pointer arithmetic, so
   *     device pointers are valid.
   * @param[in] planned_buffer_devices One entry per buffer (same count as
   *     `buffers`), indicating the `Device` (type + index) for each buffer.
   *     Different buffers can target the same device type but different
   *     indices (e.g., `cuda:0` vs `cuda:1`). For CPU-only programs, use the
   *     single-arg constructor instead.
   */
  HierarchicalAllocator(
      Span<Span<uint8_t>> buffers,
      Span<const etensor::Device> planned_buffer_devices)
      : buffers_(buffers), planned_buffer_devices_(planned_buffer_devices) {
    ET_CHECK_MSG(
        planned_buffer_devices.size() == buffers.size(),
        "planned_buffer_devices size (%" ET_PRIsize_t
        ") must match buffers size (%" ET_PRIsize_t ")",
        planned_buffer_devices.size(),
        buffers.size());
  }

  /**
   * DEPRECATED: Use spans instead.
   */
  ET_DEPRECATED HierarchicalAllocator(
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
  ET_NODISCARD Result<void*> get_offset_address(
      uint32_t memory_id,
      size_t offset_bytes,
      size_t size_bytes) {
    // Check for integer overflow in offset_bytes + size_bytes.
    size_t end_bytes = 0;
    ET_CHECK_OR_RETURN_ERROR(
        !c10::add_overflows(offset_bytes, size_bytes, &end_bytes),
        InvalidArgument,
        "Integer overflow in offset_bytes (%" ET_PRIsize_t
        ") + size_bytes (%" ET_PRIsize_t ")",
        offset_bytes,
        size_bytes);
    ET_CHECK_OR_RETURN_ERROR(
        memory_id < buffers_.size(),
        InvalidArgument,
        "id %" PRIu32 " >= %" ET_PRIsize_t,
        memory_id,
        buffers_.size());
    Span<uint8_t> buffer = buffers_[memory_id];
    ET_CHECK_OR_RETURN_ERROR(
        end_bytes <= buffer.size(),
        MemoryAllocationFailed,
        "offset_bytes (%" ET_PRIsize_t ") + size_bytes (%" ET_PRIsize_t
        ") >= allocator size (%" ET_PRIsize_t
        ") "
        "for memory_id %" PRIu32,
        offset_bytes,
        size_bytes,
        buffer.size(),
        memory_id);
    return buffer.data() + offset_bytes;
  }

  /**
   * Returns per-buffer device metadata. One entry per buffer, same count as
   * the `buffers` passed to the constructor. Each entry is a `Device`
   * carrying both type and index, so callers can distinguish e.g. `cuda:0`
   * from `cuda:1`. Empty if no device metadata was provided (CPU-only
   * program).
   */
  Span<const etensor::Device> planned_buffer_devices() const {
    return planned_buffer_devices_;
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
    for (const auto i : c10::irange(n_allocators)) {
      span_array_[i] =
          Span<uint8_t>(allocators[i].base_address(), allocators[i].size());
    }
    return {span_array_, n_allocators};
  }

  /// The underlying buffers.
  Span<Span<uint8_t>> buffers_;

  /// Per-buffer device metadata. Empty when no device info was provided
  /// (CPU-only program).
  Span<const etensor::Device> planned_buffer_devices_;
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
