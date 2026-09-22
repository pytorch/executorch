/* Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "arm_memory_allocator.h"

namespace {

// newlib-nano does not reliably handle the z length modifier. Keep allocator
// logs on the Arm runners to unsigned long with %lu.
using printf_size_t = unsigned long;

} // namespace

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
    : executorch::runtime::MemoryAllocator(size, base_address),
      cur_(base_address) {
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address, size);
#endif
}

void* ArmMemoryAllocator::allocate(size_t size, size_t alignment) {
  const uint8_t* const begin = base_address();
  const uint8_t* const end = begin == nullptr ? nullptr : begin + this->size();

  if (!begin || !end) {
    ET_LOG(Error, "allocate() on zero-capacity allocator");
    return nullptr;
  }
  if (!isPowerOf2(alignment)) {
    const printf_size_t alignment_for_log =
        static_cast<printf_size_t>(alignment);
    ET_LOG(Error, "Alignment %lu is not a power of 2", alignment_for_log);
    return nullptr;
  }

  uint8_t* start = alignPointer(cur_, alignment);
  size_t padding = static_cast<size_t>(start - cur_);
  size_t available = static_cast<size_t>(end - cur_);
  if (padding > available || size > available - padding) {
    const printf_size_t requested_for_log =
        static_cast<printf_size_t>(padding + size);
    const printf_size_t available_for_log =
        static_cast<printf_size_t>(available);
    ET_LOG(
        Error,
        "Memory allocation failed: %luB requested (adjusted for alignment), %luB available",
        requested_for_log,
        available_for_log);
    return nullptr;
  }

  uint8_t* allocated_end = start + size;
  EXECUTORCH_TRACK_ALLOCATION(prof_id(), allocated_end - cur_);
  cur_ = allocated_end;

#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_unpoison_buffer(start, size);
#endif
  return static_cast<void*>(start);
}

size_t ArmMemoryAllocator::used_size() const {
  const uint8_t* const begin = base_address();
  return begin == nullptr ? 0 : static_cast<size_t>(cur_ - begin);
}

size_t ArmMemoryAllocator::free_size() const {
  const uint8_t* const begin = base_address();
  const uint8_t* const end = begin == nullptr ? nullptr : begin + size();
  return end == nullptr ? 0 : static_cast<size_t>(end - cur_);
}

void ArmMemoryAllocator::reset() {
  cur_ = base_address();
#if defined(EXECUTORCH_ENABLE_ADDRESS_SANITIZER)
  asan_poison_buffer(base_address(), size());
#endif
}
