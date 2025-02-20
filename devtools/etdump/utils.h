// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstddef>
#include <cstdint>

#pragma once

namespace executorch {
namespace etdump {
namespace internal {

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
