#pragma once
#include <cstdint>

#include <executorch/backends/cuda/runtime/c10/macros/Macros.h>

namespace executorch::backends::cuda::c10 {

/**
 * quint4x2 is for un-signed 4 bit quantized Tensors that are packed to byte
 * boundary.
 */
struct alignas(1) quint4x2 {
  using underlying = uint8_t;
  uint8_t val_;
  quint4x2() = default;
  STANDALONE_HOST_DEVICE explicit quint4x2(uint8_t val) : val_(val) {}
};

} // namespace executorch::backends::cuda::c10
