#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>

namespace standalone::slim {

inline SlimTensor scalar_to_tensor(
    const standalone::c10::Scalar& s,
    const standalone::c10::Device& device = CPU_DEVICE) {
  SlimTensor result = empty_strided({}, {}, s.type(), device);
  result.fill_(s);
  return result;
}

} // namespace standalone::slim
