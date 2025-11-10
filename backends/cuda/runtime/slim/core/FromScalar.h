#pragma once

#include <executorch/backends/cuda/runtime/slim/core/Empty.h>

namespace executorch::backends::cuda::slim {

inline SlimTensor
scalar_to_tensor(const executorch::backends::cuda::c10::Scalar &s,
                 const executorch::backends::cuda::c10::Device &device = CPU_DEVICE) {
  SlimTensor result = empty_strided({}, {}, s.type(), device);
  result.fill_(s);
  return result;
}

} // namespace executorch::backends::cuda::slim
