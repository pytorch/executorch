/* Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/memory_allocator.h>

using executorch::runtime::MemoryAllocator;

#pragma once

// Setup our own allocator that can show some extra stuff like used and free
// memory info
class ArmMemoryAllocator : public executorch::runtime::MemoryAllocator {
 public:
  ArmMemoryAllocator(uint32_t size, uint8_t* base_address);

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override;

  // Returns the used size of the allocator's memory buffer.
  size_t used_size() const;

  // Returns the free size of the allocator's memory buffer.
  size_t free_size() const;
  void reset();

 private:
  size_t used_;
};
