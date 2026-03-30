/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/executor/dynamic_allocator.h>

#ifdef ET_DYNAMIC_ALLOCATOR_ENABLED

#include <algorithm>
#include <cstring>

#include <executorch/runtime/platform/platform.h>

namespace executorch {
namespace runtime {

/**
 * Default DynamicAllocator implementation backed by the PAL (Platform
 * Abstraction Layer). Uses et_pal_allocate/et_pal_free for memory management
 * and a 2x growth policy on reallocation to amortize allocation cost.
 */
class PalDynamicAllocator : public DynamicAllocator {
 public:
  void* allocate(size_t size, size_t alignment, size_t* actual_size) override {
    // Over-allocate to accommodate alignment and the raw-pointer bookkeeping.
    size_t alloc_size = size + sizeof(void*) + alignment;
    void* raw = pal_allocate(alloc_size);
    if (raw == nullptr) {
      return nullptr;
    }
    void* aligned = align_pointer(raw, alignment);
    // Store the raw pointer just before the aligned pointer so we can free it.
    store_raw_pointer(aligned, raw);
    std::memset(aligned, 0, size);
    if (actual_size) {
      *actual_size = size;
    }
    return aligned;
  }

  void* reallocate(
      void* ptr,
      size_t old_size,
      size_t new_size,
      size_t alignment,
      size_t* actual_size) override {
    if (ptr == nullptr) {
      return allocate(new_size, alignment, actual_size);
    }
    // Growth policy: at least 2x old_size to amortize repeated resizes.
    size_t target = std::max(new_size, old_size * 2);
    size_t alloc_size = target + sizeof(void*) + alignment;
    void* raw = pal_allocate(alloc_size);
    if (raw == nullptr) {
      return nullptr;
    }
    void* aligned = align_pointer(raw, alignment);
    store_raw_pointer(aligned, raw);
    // Copy old data and zero-initialize the new region.
    size_t copy_size = std::min(old_size, new_size);
    if (copy_size > 0) {
      std::memcpy(aligned, ptr, copy_size);
    }
    if (target > copy_size) {
      std::memset(
          static_cast<uint8_t*>(aligned) + copy_size, 0, target - copy_size);
    }
    // Free old allocation.
    free(ptr);
    if (actual_size) {
      *actual_size = target;
    }
    return aligned;
  }

  void free(void* ptr) override {
    if (ptr == nullptr) {
      return;
    }
    void* raw = load_raw_pointer(ptr);
    pal_free(raw);
  }

 private:
  static void* align_pointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    // Reserve space for the raw pointer bookkeeping.
    addr += sizeof(void*);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
  }

  static void store_raw_pointer(void* aligned, void* raw) {
    // Store the raw (unaligned) pointer immediately before the aligned pointer.
    reinterpret_cast<void**>(aligned)[-1] = raw;
  }

  static void* load_raw_pointer(void* aligned) {
    return reinterpret_cast<void**>(aligned)[-1];
  }
};

} // namespace runtime
} // namespace executorch

#endif // ET_DYNAMIC_ALLOCATOR_ENABLED
