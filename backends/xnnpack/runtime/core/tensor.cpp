#include <executorch/backends/xnnpack/runtime/core/tensor.h>

#include <c10/util/safe_numerics.h>
#include <executorch/runtime/platform/log.h>

#include <cstdlib>

namespace executorch::backends::xnnpack::core {

using executorch::runtime::Span;

Storage::~Storage() {
  if (owner == StorageOwner::Self) {
    std::free(data);
  }
}

Storage::Storage(Storage&& other) noexcept
    : data(other.data), owner(other.owner), size_in_bytes(other.size_in_bytes) {
  other.data = nullptr;
  other.owner = StorageOwner::External;
  other.size_in_bytes = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
  if (this != &other) {
    if (owner == StorageOwner::Self) {
      std::free(data);
    }
    data = other.data;
    owner = other.owner;
    size_in_bytes = other.size_in_bytes;
    other.data = nullptr;
    other.owner = StorageOwner::External;
    other.size_in_bytes = 0;
  }
  return *this;
}

runtime::Result<Storage> Storage::create_owned(size_t size_in_bytes) {
  void* data = std::malloc(size_in_bytes);
  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr || size_in_bytes == 0,
      MemoryAllocationFailed,
      "Failed to allocate %zu bytes for tensor storage",
      size_in_bytes);

  Storage s;
  s.data = data;
  s.owner = StorageOwner::Self;
  s.size_in_bytes = size_in_bytes;
  return s;
}

namespace {
runtime::Result<size_t> checked_num_elements(Span<const uint64_t> sizes) {
  size_t num_elements = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    size_t next;
    ET_CHECK_OR_RETURN_ERROR(
        !c10::mul_overflows(num_elements, static_cast<size_t>(sizes[i]), &next),
        InvalidArgument,
        "Overflow computing number of elements at dimension %zu",
        i);
    num_elements = next;
  }
  return num_elements;
}
} // namespace

runtime::Result<size_t> Tensor::numel() const {
  return checked_num_elements({sizes.data(), sizes.size()});
}

runtime::Error Tensor::resize(std::vector<uint64_t> new_sizes) {
  ET_UNWRAP(
      new_size_in_bytes,
      compute_storage_size({new_sizes.data(), new_sizes.size()}, dtype));

  if (new_size_in_bytes <= storage.size_in_bytes) {
    sizes = std::move(new_sizes);
    return runtime::Error::Ok;
  }

  ET_CHECK_OR_RETURN_ERROR(
      storage.owner == StorageOwner::Self,
      NotSupported,
      "Cannot grow storage of a non-owned tensor");

  void* new_data = std::realloc(storage.data, new_size_in_bytes);
  ET_CHECK_OR_RETURN_ERROR(
      new_data != nullptr,
      MemoryAllocationFailed,
      "Failed to reallocate %zu bytes during resize",
      new_size_in_bytes);

  storage.data = new_data;
  storage.size_in_bytes = new_size_in_bytes;
  sizes = std::move(new_sizes);
  return runtime::Error::Ok;
}

runtime::Result<size_t> compute_storage_size(
    Span<const uint64_t> sizes,
    DType dtype) {
  ET_UNWRAP(num_elements, checked_num_elements(sizes));

  switch (dtype) {
    case DType::Int64:
    case DType::UInt64: {
      size_t bytes;
      ET_CHECK_OR_RETURN_ERROR(
          !c10::mul_overflows(num_elements, size_t{8}, &bytes),
          InvalidArgument,
          "Overflow computing storage size in bytes");
      return bytes;
    }
    case DType::Float32:
    case DType::QInt32: {
      size_t bytes;
      ET_CHECK_OR_RETURN_ERROR(
          !c10::mul_overflows(num_elements, size_t{4}, &bytes),
          InvalidArgument,
          "Overflow computing storage size in bytes");
      return bytes;
    }
    case DType::Float16:
    case DType::BFloat16: {
      size_t bytes;
      ET_CHECK_OR_RETURN_ERROR(
          !c10::mul_overflows(num_elements, size_t{2}, &bytes),
          InvalidArgument,
          "Overflow computing storage size in bytes");
      return bytes;
    }
    case DType::QInt8:
    case DType::QUInt8:
      return num_elements;
    case DType::QInt4:
      // Two 4-bit elements per byte, rounded up (written to avoid overflow
      // in the round-up).
      return num_elements / 2 + (num_elements % 2);
  }

  ET_LOG(
      Error,
      "Unknown DType %d in compute_storage_size",
      static_cast<int>(dtype));
  return runtime::Error::InvalidArgument;
}

} // namespace executorch::backends::xnnpack::core
