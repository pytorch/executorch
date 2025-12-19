#pragma once
#include <cstdint>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>

namespace standalone::c10 {

/**
 * quint8 is for unsigned 8 bit quantized Tensors
 */
struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  quint8() = default;
  STANDALONE_HOST_DEVICE explicit quint8(uint8_t val) : val_(val) {}
};

} // namespace standalone::c10
