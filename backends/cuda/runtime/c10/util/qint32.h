#pragma once
#include <cstdint>

#include <executorch/backends/cuda/runtime/c10/macros/Macros.h>

namespace executorch::backends::cuda::c10 {

/**
 * qint32 is for signed 32 bit quantized Tensors
 */
struct alignas(4) qint32 {
  using underlying = int32_t;
  int32_t val_;
  qint32() = default;
  STANDALONE_HOST_DEVICE explicit qint32(int32_t val) : val_(val) {}
};

} // namespace executorch::backends::cuda::c10
