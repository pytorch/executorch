#include <executorch/backends/xnnpack/runtime/core/tensor.h>

#include <cstdlib>
#include <numeric>

namespace executorch::backends::xnnpack::core {

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

Storage Storage::create_owned(size_t size_in_bytes) {
    Storage s;
    s.data = std::malloc(size_in_bytes);
    s.owner = StorageOwner::Self;
    s.size_in_bytes = size_in_bytes;
    return s;
}

size_t Tensor::numel() const {
    return std::accumulate(
        sizes.begin(), sizes.end(), size_t{1}, std::multiplies<>());
}

bool Tensor::resize(std::vector<uint64_t> new_sizes) {
    size_t new_size_in_bytes = compute_storage_size(new_sizes, dtype);

    if (new_size_in_bytes <= storage.size_in_bytes) {
        sizes = std::move(new_sizes);
        return true;
    }

    if (storage.owner != StorageOwner::Self) {
        return false;
    }

    void* new_data = std::realloc(storage.data, new_size_in_bytes);
    if (!new_data) { return false; }

    storage.data = new_data;
    storage.size_in_bytes = new_size_in_bytes;
    sizes = std::move(new_sizes);
    return true;
}

size_t compute_storage_size(Span<const uint64_t> sizes, DType dtype) {
    size_t num_elements = std::accumulate(
        sizes.begin(), sizes.end(), size_t{1}, std::multiplies<>());

    switch (dtype) {
        case DType::Float32:    return num_elements * 4;
        case DType::QInt8Sym:   return num_elements * 1;
        case DType::QUInt8Asym: return num_elements * 1;
        case DType::QInt4Sym:   return (num_elements + 1) / 2;
        case DType::QInt32Sym:  return num_elements * 4;
    }
}

}
