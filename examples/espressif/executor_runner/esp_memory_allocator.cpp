/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "esp_memory_allocator.h"

EspMemoryAllocator::EspMemoryAllocator(uint32_t size, uint8_t* base_address)
    : MemoryAllocator(size, base_address), used_(0) {}

void* EspMemoryAllocator::allocate(size_t size, size_t alignment) {
  void* ret = executorch::runtime::MemoryAllocator::allocate(size, alignment);
  if (ret != nullptr) {
    // Align with the same code as in MemoryAllocator::allocate() to keep
    // used_ "in sync". As alignment is expected to be power of 2 (checked by
    // MemoryAllocator::allocate()) we can check if the lower bits
    // (same as alignment - 1) is zero or not.
    if ((size & (alignment - 1)) == 0) {
      // Already aligned.
      used_ += size;
    } else {
      used_ = (used_ | (alignment - 1)) + 1 + size;
    }
  }
  return ret;
}

size_t EspMemoryAllocator::used_size() const {
  return used_;
}

size_t EspMemoryAllocator::free_size() const {
  return executorch::runtime::MemoryAllocator::size() - used_;
}

void EspMemoryAllocator::reset() {
  executorch::runtime::MemoryAllocator::reset();
  used_ = 0;
}
