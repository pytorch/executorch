#pragma once

#include <executorch/compiler/Compiler.h>
#include <executorch/core/Assert.h>
#include <executorch/core/Log.h>
#include <executorch/core/Result.h>
#include <executorch/executor/MemoryAllocator.h>
#include <cstdint>

namespace torch {
namespace executor {

// A group of allocators that can be used to represent a device's memory
// hierarchy.
class HierarchicalAllocator {
 public:
  // Constructs a new hierarchycal allocator with the given array of allocators.
  // Memory IDs are assigned based on the index in the 'allocators' array. E.g.
  // the first allocator in the array will have a memory ID of 0.
  HierarchicalAllocator(uint32_t n_allocators, MemoryAllocator* allocators)
      : n_allocators_(n_allocators), allocators_(allocators) {}

  /**
   * Returns the address at the byte offset `offset_bytes` from the given
   * allocator's base address, which should have at least `size_bytes` of memory
   * available inside the allocator's buffer.
   *
   * This is useful to point an object to this address when such information has
   * been predetermined. This method assumes that the given memory's allocator
   * has already reserved enough memory (i.e. there's no actual allocation call
   * to the underlying memory allocator).
   *
   * @param[in] memory_id The ID of the allocator in the hierarchy.
   * @param[in] offset_bytes The offset in bytes into the memory of the
   *     specified allocator.
   * @param[in] size_bytes The amount of memory that should be available at
   *     the offset.
   *
   * @returns On success, the address of the requested byte offset into the
   *     specified allocator. On failure, a non-Ok Error.
   */
  __ET_NODISCARD Result<void*> get_offset_address(
      uint32_t memory_id,
      size_t offset_bytes,
      size_t size_bytes) {
    Result<MemoryAllocator*> allocator_result = get_allocator(memory_id);
    if (!allocator_result.ok()) {
      return allocator_result.error();
    }
    auto allocator = allocator_result.get();
    ET_CHECK_OR_RETURN_ERROR(
        offset_bytes + size_bytes <= allocator->size(),
        MemoryAllocationFailed,
        "offset_bytes (%zu) + size_bytes (%zu) >= allocator size (%" PRIu32
        ") for memory_id %" PRIu32,
        offset_bytes,
        size_bytes,
        allocator->size(),
        memory_id);
    return allocator->base_address() + offset_bytes;
  }

  virtual ~HierarchicalAllocator() {}

 private:
  /// Returns the memory allocator for the given 'memory_id' in the hierarchy.
  Result<MemoryAllocator*> get_allocator(uint32_t memory_id) const {
    ET_CHECK_OR_RETURN_ERROR(
        memory_id < n_allocators_,
        InvalidArgument,
        "Memory id %" PRIu32 " >= n_allocators_ %" PRIu32,
        memory_id,
        n_allocators_);
    return &allocators_[memory_id];
  }

  // The HierarchicalAllocator holds n_allocators_ MemoryAllocators.
  uint32_t n_allocators_;

  // Memory allocators as an array, each ID corresponds to their index.
  MemoryAllocator* allocators_;
};
} // namespace executor
} // namespace torch
