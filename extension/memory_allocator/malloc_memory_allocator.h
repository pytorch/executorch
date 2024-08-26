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
#include <vector>

#include <executorch/runtime/core/memory_allocator.h>

namespace executorch {
namespace extension {

/**
 * Dynamically allocates memory using malloc() and frees all pointers at
 * destruction time.
 *
 * For systems with malloc(), this can be easier than using a fixed-sized
 * MemoryAllocator.
 */
class MallocMemoryAllocator : public executorch::runtime::MemoryAllocator {
 public:
  /**
   * Construct a new Malloc memory allocator via an optional alignment size
   * parameter.
   *
   * @param[in] align_size An optional parameter to specify alignment parameter
   * for each allocate() call.
   */
  MallocMemoryAllocator() : MemoryAllocator(0, nullptr) {}

  ~MallocMemoryAllocator() override {
    reset();
  }

  /**
   * Allocates 'size' bytes of memory, returning a pointer to the allocated
   * region, or nullptr upon failure. The size will be rounded up based on the
   * memory alignment size.
   */
  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    EXECUTORCH_TRACK_ALLOCATION(prof_id(), size);

    if (!isPowerOf2(alignment)) {
      ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
      return nullptr;
    }

    // The minimum alignment that malloc() is guaranteed to provide.
    static constexpr size_t kMallocAlignment = alignof(std::max_align_t);
    if (alignment > kMallocAlignment) {
      // To get higher alignments, allocate extra and then align the returned
      // pointer. This will waste an extra `alignment` bytes every time, but
      // this is the only portable way to get aligned memory from the heap.
      size += alignment;
    }
    mem_ptrs_.emplace_back(std::malloc(size));
    return alignPointer(mem_ptrs_.back(), alignment);
  }

  // Free up each hosted memory pointer. The memory was created via malloc.
  void reset() override {
    for (auto mem_ptr : mem_ptrs_) {
      free(mem_ptr);
    }
    mem_ptrs_.clear();
  }

 private:
  std::vector<void*> mem_ptrs_;
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::MallocMemoryAllocator;
} // namespace util
} // namespace executor
} // namespace torch
