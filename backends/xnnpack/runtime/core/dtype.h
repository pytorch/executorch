#pragma once

namespace executorch::backends::xnnpack::core {

enum class DType {
  // Floating point
  Float32,
  Float16,
  BFloat16,

  // Non-quantized integer
  Int64,
  UInt64,

  // Quantized — signed
  QInt8,
  QInt4,
  QInt32,

  // Quantized — unsigned
  QUInt8,
};

} // namespace executorch::backends::xnnpack::core
