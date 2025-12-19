#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>

namespace executorch::backends::aoti::slim {
inline SlimTensor zeros(
    executorch::backends::aoti::slim::c10::IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  SlimTensor tensor = empty(sizes, dtype, device);
  tensor.fill_(executorch::backends::aoti::slim::c10::Scalar(0));
  return tensor;
}

inline SlimTensor zeros_like(const SlimTensor& other) {
  return zeros(other.sizes(), other.dtype(), other.device());
}

inline SlimTensor ones(
    executorch::backends::aoti::slim::c10::IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  SlimTensor tensor = empty(sizes, dtype, device);
  tensor.fill_(executorch::backends::aoti::slim::c10::Scalar(1));
  return tensor;
}

inline SlimTensor ones_like(const SlimTensor& other) {
  return ones(other.sizes(), other.dtype(), other.device());
}

} // namespace executorch::backends::aoti::slim
