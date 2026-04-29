/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/memory_allocator.h>

/**
 * Custom allocator for Espressif ESP32/ESP32-S3 targets that tracks
 * used and free memory. Extends the ExecuTorch MemoryAllocator with
 * additional instrumentation useful for memory-constrained embedded
 * environments.
 */
class EspMemoryAllocator : public executorch::runtime::MemoryAllocator {
 public:
  EspMemoryAllocator(uint32_t size, uint8_t* base_address);

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override;

  /// Returns the used size of the allocator's memory buffer.
  size_t used_size() const;

  /// Returns the free size of the allocator's memory buffer.
  size_t free_size() const;

  /// Resets the allocator to its initial state.
  void reset();

 private:
  size_t used_;
};
