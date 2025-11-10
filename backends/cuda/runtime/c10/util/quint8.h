#pragma once
#include <cstdint>

#include <executorch/backends/cuda/runtime/c10/macros/Macros.h>

namespace executorch::backends::cuda::c10 {

/**
 * quint8 is for unsigned 8 bit quantized Tensors
 */
struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  quint8() = default;
  STANDALONE_HOST_DEVICE explicit quint8(uint8_t val) : val_(val) {}
};

} // namespace executorch::backends::cuda::c10
