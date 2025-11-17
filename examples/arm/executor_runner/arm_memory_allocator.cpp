/* Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "arm_memory_allocator.h"

#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
extern "C" {
void __asan_poison_memory_region(void* addr, size_t size);
void __asan_unpoison_memory_region(void* addr, size_t size);
}

static void asan_poison_buffer(uint8_t* base, size_t size) {
  if (base != nullptr && size > 0) {
    __asan_poison_memory_region(base, size);
  }
}

static void asan_unpoison_buffer(void* base, size_t size) {
  if (base != nullptr && size > 0) {
    __asan_unpoison_memory_region(base, size);
  }
}
#endif

ArmMemoryAllocator::ArmMemoryAllocator(uint32_t size, uint8_t* base_address)
    : MemoryAllocator(size, base_address), used_(0) {
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address, size);
#endif
}

void* ArmMemoryAllocator::allocate(size_t size, size_t alignment) {
  void* ret = executorch::runtime::MemoryAllocator::allocate(size, alignment);
  if (ret != nullptr) {
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
    asan_unpoison_buffer(ret, size);
#endif
    // Align with the same code as in MemoryAllocator::allocate() to keep
    // used_ "in sync" As alignment is expected to be power of 2 (checked by
    // MemoryAllocator::allocate()) we can check it the lower bits
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

size_t ArmMemoryAllocator::used_size() const {
  return used_;
}

size_t ArmMemoryAllocator::free_size() const {
  return executorch::runtime::MemoryAllocator::size() - used_;
}

void ArmMemoryAllocator::reset() {
  executorch::runtime::MemoryAllocator::reset();
  used_ = 0;
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address(), size());
#endif
}
