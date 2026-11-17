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
// all of it "internal". With ESPNN_HOST_PSRAM, PSRAM allocations come from the
// same heap; otherwise there is no PSRAM. ESPNN_HOST_INTERNAL_BYTES in the
// environment limits the internal aligned allocations to that many bytes.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

#define MALLOC_CAP_8BIT (1U << 2)
#define MALLOC_CAP_SPIRAM (1U << 10)
#define MALLOC_CAP_INTERNAL (1U << 11)

#ifdef ESPNN_HOST_PSRAM
constexpr bool kHostPsram = true;
#else
constexpr bool kHostPsram = false;
#endif

namespace espnn_host {

inline std::unordered_map<void*, size_t> g_internal_blocks;
inline size_t g_internal_used = 0;

inline size_t internal_free() {
  static const char* limit = std::getenv("ESPNN_HOST_INTERNAL_BYTES");
  const size_t capacity =
      limit != nullptr ? std::strtoull(limit, nullptr, 10) : SIZE_MAX / 2;
  return capacity > g_internal_used ? capacity - g_internal_used : 0;
}

} // namespace espnn_host

inline void*
heap_caps_aligned_alloc(size_t alignment, size_t size, uint32_t caps) {
  const bool psram = (caps & MALLOC_CAP_SPIRAM) != 0;
  if ((psram && !kHostPsram) || size == 0 ||
      (!psram && size > espnn_host::internal_free()))
    return nullptr;
  // Exactly size bytes, so that AddressSanitizer sees any overrun.
  void* memory = nullptr;
  if (posix_memalign(&memory, alignment, size) != 0)
    return nullptr;
  if (!psram) {
    espnn_host::g_internal_blocks[memory] = size;
    espnn_host::g_internal_used += size;
  }
  return memory;
}

inline void* heap_caps_calloc(size_t count, size_t size, uint32_t) {
  return std::calloc(count, size);
}

inline void heap_caps_free(void* memory) {
  const auto block = espnn_host::g_internal_blocks.find(memory);
  if (block != espnn_host::g_internal_blocks.end()) {
    espnn_host::g_internal_used -= block->second;
    espnn_host::g_internal_blocks.erase(block);
  }
  std::free(memory);
}

inline size_t heap_caps_get_largest_free_block(uint32_t caps) {
  if ((caps & MALLOC_CAP_SPIRAM) != 0)
    return kHostPsram ? SIZE_MAX / 2 : 0;
  return espnn_host::internal_free();
}

inline size_t heap_caps_get_total_size(uint32_t caps) {
  return heap_caps_get_largest_free_block(caps);
}
