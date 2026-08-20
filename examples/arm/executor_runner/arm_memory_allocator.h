/* Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/memory_allocator.h>

// Custom allocator that keeps Arm-specific logging format strings out of the
// generic runtime allocator path and poisons/unpoisons its buffer for
// AddressSanitizer.
class ArmMemoryAllocator : public executorch::runtime::MemoryAllocator {
 public:
  ArmMemoryAllocator(uint32_t size, uint8_t* base_address);

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override;

  size_t used_size() const override;

  size_t free_size() const override;

  void reset() override;

 private:
  uint8_t* cur_;
};
