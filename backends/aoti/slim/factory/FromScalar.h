#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>

namespace executorch::backends::aoti::slim {

inline SlimTensor scalar_to_tensor(
    const executorch::backends::aoti::slim::c10::Scalar& s,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  SlimTensor result = empty_strided({}, {}, s.type(), device);
  result.fill_(s);
  return result;
}

} // namespace executorch::backends::aoti::slim
