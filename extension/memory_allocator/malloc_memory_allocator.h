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
#include <cstdlib>
#include <vector>

#include <executorch/extension/memory_allocator/memory_allocator_utils.h>
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
    if (!isPowerOf2(alignment)) {
      ET_LOG(Error, "Alignment %zu is not a power of 2", alignment);
      return nullptr;
    }

    auto adjusted_size_value =
        executorch::extension::utils::get_aligned_size(size, alignment);
    if (!adjusted_size_value.ok()) {
      return nullptr;
    }
    size = adjusted_size_value.get();
    void* mem_ptr = std::malloc(size);
    if (!mem_ptr) {
      ET_LOG(Error, "Malloc failed to allocate %zu bytes", size);
      return nullptr;
    }
    mem_ptrs_.emplace_back(mem_ptr);
    EXECUTORCH_TRACK_ALLOCATION(prof_id(), size);
    return alignPointer(mem_ptrs_.back(), alignment);
  }

  // Free up each hosted memory pointer. The memory was created via malloc.
  void reset() override {
    for (auto mem_ptr : mem_ptrs_) {
      std::free(mem_ptr);
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
