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
    // Keep used_ in sync with the underlying MemoryAllocator by computing it
    // from the returned pointer and requested size, which implicitly includes
    // any padding/alignment the base allocator applied.
    uint8_t* end_ptr = static_cast<uint8_t*>(ret) + size;
    used_ = static_cast<size_t>(end_ptr - base_address());
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
