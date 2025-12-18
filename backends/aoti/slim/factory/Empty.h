#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>

namespace standalone::slim {
// The returned SlimTensor owns the underlying storage
inline SlimTensor empty_strided(
    standalone::c10::IntArrayRef sizes,
    standalone::c10::IntArrayRef strides,
    standalone::c10::ScalarType dtype,
    const standalone::c10::Device& device = CPU_DEVICE) {
  Storage storage = new_storage(sizes, strides, dtype, device);
  return SlimTensor(std::move(storage), sizes, strides, dtype, 0);
}

inline SlimTensor empty(
    standalone::c10::IntArrayRef sizes,
    standalone::c10::ScalarType dtype,
    const standalone::c10::Device& device = CPU_DEVICE) {
  std::vector<int64_t> contig_strides =
      standalone::slim::compute_contiguous_strides(sizes);
  Storage storage = new_storage(sizes, contig_strides, dtype, device);
  return SlimTensor(std::move(storage), sizes, contig_strides, dtype, 0);
}

inline SlimTensor empty_like(const SlimTensor& other) {
  return empty_strided(
      other.sizes(), other.strides(), other.dtype(), other.device());
}
} // namespace standalone::slim
