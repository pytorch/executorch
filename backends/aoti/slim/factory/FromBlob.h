#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>

namespace standalone::slim {

// The returned SlimTensor does not own the underlying storage
inline SlimTensor from_blob(
    void* data,
    standalone::c10::IntArrayRef sizes,
    standalone::c10::IntArrayRef strides,
    standalone::c10::ScalarType dtype,
    const standalone::c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  STANDALONE_CHECK(data != nullptr, "data pointer can not be nullptr");

  Storage storage(new MaybeOwningStorage(
      device,
      data,
      compute_storage_nbytes(
          sizes, strides, elementSize(dtype), storage_offset)));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}

inline SlimTensor from_blob(
    void* data,
    standalone::c10::IntArrayRef sizes,
    standalone::c10::ScalarType dtype,
    const standalone::c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  std::vector<int64_t> contig_strides =
      standalone::slim::compute_contiguous_strides(sizes);
  return from_blob(data, sizes, contig_strides, dtype, device, storage_offset);
}

} // namespace standalone::slim
