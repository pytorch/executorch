#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <executorch/backends/cuda/runtime/slim/core/SlimTensor.h>
#include <executorch/backends/cuda/runtime/slim/util/SizeUtil.h>

namespace executorch::backends::cuda::slim {
// The returned SlimTensor owns the underlying storage
inline SlimTensor
empty_strided(executorch::aten::IntArrayRef sizes,
              executorch::aten::IntArrayRef strides,
              executorch::backends::cuda::c10::ScalarType dtype,
              const executorch::backends::cuda::c10::Device &device = CPU_DEVICE) {
  Storage storage = new_storage(sizes, strides, dtype, device);
  return SlimTensor(std::move(storage), sizes, strides, dtype, 0);
}

inline SlimTensor empty(executorch::aten::IntArrayRef sizes,
                       executorch::backends::cuda::c10::ScalarType dtype,
                       const executorch::backends::cuda::c10::Device& device) {
  std::vector<int64_t> contig_strides =
      executorch::backends::cuda::slim::compute_contiguous_strides(et_to_c10(sizes));
  Storage storage = new_storage(sizes, vec_to_et(contig_strides), dtype, device);
  return SlimTensor(std::move(storage), sizes, vec_to_et(contig_strides), dtype, 0);
}

inline SlimTensor empty_like(const SlimTensor &other) {
  return empty_strided(other.sizes(), other.strides(), other.dtype(),
                       other.device());
}
} // namespace executorch::backends::cuda::slim
