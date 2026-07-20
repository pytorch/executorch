/* Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/memory_allocator.h>

using executorch::runtime::MemoryAllocator;

#pragma once

// Custom allocator that poisons/unpoisons its buffer for AddressSanitizer. The
// used and free byte counts are reported by the base MemoryAllocator's
// used_size() / free_size().
class ArmMemoryAllocator : public executorch::runtime::MemoryAllocator {
 public:
  ArmMemoryAllocator(uint32_t size, uint8_t* base_address);

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override;

  void reset() override;
};
