#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>

namespace standalone::slim {
inline SlimTensor zeros(
    standalone::c10::IntArrayRef sizes,
    standalone::c10::ScalarType dtype,
    const standalone::c10::Device& device = CPU_DEVICE) {
  SlimTensor tensor = empty(sizes, dtype, device);
  tensor.fill_(standalone::c10::Scalar(0));
  return tensor;
}

inline SlimTensor zeros_like(const SlimTensor& other) {
  return zeros(other.sizes(), other.dtype(), other.device());
}

inline SlimTensor ones(
    standalone::c10::IntArrayRef sizes,
    standalone::c10::ScalarType dtype,
    const standalone::c10::Device& device = CPU_DEVICE) {
  SlimTensor tensor = empty(sizes, dtype, device);
  tensor.fill_(standalone::c10::Scalar(1));
  return tensor;
}

inline SlimTensor ones_like(const SlimTensor& other) {
  return ones(other.sizes(), other.dtype(), other.device());
}

} // namespace standalone::slim
