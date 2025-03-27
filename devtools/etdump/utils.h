/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstddef>
#include <cstdint>

#pragma once

namespace executorch {
namespace etdump {
namespace internal {

/**
 * Aligns a pointer to the next multiple of `alignment`.
 *
 * @param[in] ptr Pointer to align.
 * @param[in] alignment Alignment to align to. Must be a power of 2 and cannot
 * be 0.
 *
 * @returns A pointer aligned to `alignment`.
 */
inline uint8_t* align_pointer(void* ptr, size_t alignment) {
  intptr_t addr = reinterpret_cast<intptr_t>(ptr);
  if ((addr & (alignment - 1)) == 0) {
    // Already aligned.
    return reinterpret_cast<uint8_t*>(ptr);
  }
  addr = (addr | (alignment - 1)) + 1;
  return reinterpret_cast<uint8_t*>(addr);
}

} // namespace internal
} // namespace etdump
} // namespace executorch
