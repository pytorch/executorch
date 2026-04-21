/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstring>

#include <executorch/runtime/executor/dynamic_allocator.h>
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
    size_t alloc_size = size + kHeaderSize + alignment;
    void* raw = pal_allocate(alloc_size);
    if (raw == nullptr) {
      return nullptr;
    }
    void* aligned = align_pointer(raw, alignment);
    store_header(aligned, raw, size);
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
    size_t current_capacity = load_capacity(ptr);
    if (new_size <= current_capacity) {
      if (actual_size) {
        *actual_size = current_capacity;
      }
      return ptr;
    }
    // Growth policy: at least 2x current capacity to amortize repeated resizes.
    size_t target = std::max(new_size, current_capacity * 2);
    size_t alloc_size = target + kHeaderSize + alignment;
    void* raw = pal_allocate(alloc_size);
    if (raw == nullptr) {
      return nullptr;
    }
    void* aligned = align_pointer(raw, alignment);
    store_header(aligned, raw, target);
    size_t copy_size = std::min(old_size, new_size);
    if (copy_size > 0) {
      std::memcpy(aligned, ptr, copy_size);
    }
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

  size_t allocated_size(void* ptr) const override {
    if (ptr == nullptr) {
      return 0;
    }
    return load_capacity(ptr);
  }

 private:
  static constexpr size_t kHeaderSize = sizeof(void*) + sizeof(size_t);

  static void* align_pointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    addr += kHeaderSize;
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
  }

  static void store_header(void* aligned, void* raw, size_t capacity) {
    reinterpret_cast<void**>(aligned)[-1] = raw;
    reinterpret_cast<size_t*>(
        static_cast<char*>(aligned) - sizeof(void*) - sizeof(size_t))[0] =
        capacity;
  }

  static void* load_raw_pointer(void* aligned) {
    return reinterpret_cast<void**>(aligned)[-1];
  }

  static size_t load_capacity(void* aligned) {
    return reinterpret_cast<size_t*>(
        static_cast<char*>(aligned) - sizeof(void*) - sizeof(size_t))[0];
  }
};

} // namespace runtime
} // namespace executorch
