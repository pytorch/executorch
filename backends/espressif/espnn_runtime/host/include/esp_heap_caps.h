/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Host stand-in for the ESP-IDF heap API the ESP-NN delegate uses: one heap,
// all of it "internal", no PSRAM.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#define MALLOC_CAP_8BIT (1U << 2)
#define MALLOC_CAP_SPIRAM (1U << 10)
#define MALLOC_CAP_INTERNAL (1U << 11)

inline void*
heap_caps_aligned_alloc(size_t alignment, size_t size, uint32_t caps) {
  if ((caps & MALLOC_CAP_SPIRAM) != 0 || size == 0)
    return nullptr;
  return std::aligned_alloc(
      alignment, (size + alignment - 1) & ~(alignment - 1));
}

inline void* heap_caps_calloc(size_t count, size_t size, uint32_t) {
  return std::calloc(count, size);
}

inline void heap_caps_free(void* memory) {
  std::free(memory);
}

inline size_t heap_caps_get_largest_free_block(uint32_t caps) {
  return (caps & MALLOC_CAP_SPIRAM) != 0 ? 0 : SIZE_MAX / 2;
}

inline size_t heap_caps_get_total_size(uint32_t caps) {
  return heap_caps_get_largest_free_block(caps);
}
