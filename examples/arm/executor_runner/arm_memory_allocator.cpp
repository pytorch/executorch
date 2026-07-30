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
    : MemoryAllocator(size, base_address) {
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address, size);
#endif
}

void* ArmMemoryAllocator::allocate(size_t size, size_t alignment) {
  void* ret = executorch::runtime::MemoryAllocator::allocate(size, alignment);
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  if (ret != nullptr) {
    asan_unpoison_buffer(ret, size);
  }
#endif
  return ret;
}

void ArmMemoryAllocator::reset() {
  executorch::runtime::MemoryAllocator::reset();
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address(), size());
#endif
}
