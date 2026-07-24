#pragma once

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace executorch::backends::xnnpack::core {

enum class StorageOwner { Arena, External, Self };

struct Storage {
  void* data = nullptr;
  StorageOwner owner = StorageOwner::External;
  size_t size_in_bytes = 0;

  Storage() = default;
  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  Storage(Storage&& other) noexcept;
  Storage& operator=(Storage&& other) noexcept;

  static runtime::Result<Storage> create_owned(size_t size_in_bytes);
};

struct Tensor {
  DType dtype;
  std::vector<uint64_t> sizes;
  Storage storage;
  std::vector<Storage> aux_storage;

  template <class T>
  const T* data_const() const {
    return static_cast<const T*>(storage.data);
  }
  template <class T>
  T* data_mut() {
    return static_cast<T*>(storage.data);
  }

  runtime::Result<size_t> numel() const;
  runtime::Error resize(std::vector<uint64_t> new_sizes);
};

runtime::Result<size_t> compute_storage_size(
    runtime::Span<const uint64_t> sizes,
    DType dtype);

} // namespace executorch::backends::xnnpack::core
