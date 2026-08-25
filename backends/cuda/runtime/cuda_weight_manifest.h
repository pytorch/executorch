/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda {

constexpr char kCudaFqnWeightsMagic[] = "ETCUDAFQN3";
constexpr size_t kCudaFqnWeightsMagicSize = sizeof(kCudaFqnWeightsMagic) - 1;

struct CudaFqnWeightEntry {
  std::string fqn;
  std::string storage_key;
  uint64_t storage_nbytes{0};
  int32_t dtype{0};
  int32_t device_type{0};
  int64_t storage_offset{0};
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct CudaFqnWeightManifest {
  std::string so_blob_key;
  std::vector<CudaFqnWeightEntry> entries;
};

inline bool is_supported_cuda_fqn_dtype(int32_t dtype) {
  // Values match c10::ScalarType and the slim AOTI runtime.
  switch (dtype) {
    case 0: // Byte
    case 1: // Char
    case 2: // Short
    case 3: // Int
    case 4: // Long
    case 5: // Half
    case 6: // Float
    case 11: // Bool
    case 15: // BFloat16
      return true;
    default:
      return false;
  }
}

inline bool is_supported_cuda_fqn_device_type(int32_t device_type) {
  // Values match c10::DeviceType and the slim AOTI runtime.
  return device_type == 0 || device_type == 1; // CPU or CUDA
}

inline bool is_cuda_fqn_weight_manifest(const void* data, size_t size) {
  return data != nullptr && size >= kCudaFqnWeightsMagicSize &&
      std::memcmp(data, kCudaFqnWeightsMagic, kCudaFqnWeightsMagicSize) == 0;
}

namespace detail {

class CudaWeightManifestReader final {
 public:
  CudaWeightManifestReader(const void* data, size_t size)
      : cursor_(static_cast<const uint8_t*>(data)), end_(cursor_ + size) {}

  bool skip(size_t size) {
    if (remaining() < size) {
      return false;
    }
    cursor_ += size;
    return true;
  }

  bool read_u32(uint32_t& value) {
    uint64_t wide = 0;
    if (!read_unsigned(wide, 4)) {
      return false;
    }
    value = static_cast<uint32_t>(wide);
    return true;
  }

  bool read_i32(int32_t& value) {
    uint32_t raw = 0;
    if (!read_u32(raw)) {
      return false;
    }
    std::memcpy(&value, &raw, sizeof(value));
    return true;
  }

  bool read_u64(uint64_t& value) {
    return read_unsigned(value, 8);
  }

  bool read_i64(int64_t& value) {
    uint64_t raw = 0;
    if (!read_u64(raw)) {
      return false;
    }
    std::memcpy(&value, &raw, sizeof(value));
    return true;
  }

  bool read_string(std::string& value) {
    uint32_t size = 0;
    if (!read_u32(size) || remaining() < size) {
      return false;
    }
    value.assign(reinterpret_cast<const char*>(cursor_), size);
    cursor_ += size;
    return true;
  }

  bool empty() const {
    return cursor_ == end_;
  }

 private:
  size_t remaining() const {
    return static_cast<size_t>(end_ - cursor_);
  }

  bool read_unsigned(uint64_t& value, size_t width) {
    if (remaining() < width) {
      return false;
    }
    value = 0;
    for (size_t index = 0; index < width; ++index) {
      value |= static_cast<uint64_t>(cursor_[index]) << (index * 8);
    }
    cursor_ += width;
    return true;
  }

  const uint8_t* cursor_;
  const uint8_t* end_;
};

} // namespace detail

inline executorch::runtime::Error parse_cuda_fqn_weight_manifest(
    const void* data,
    size_t size,
    CudaFqnWeightManifest& manifest) {
  using executorch::runtime::Error;
  if (!is_cuda_fqn_weight_manifest(data, size)) {
    return Error::InvalidProgram;
  }

  detail::CudaWeightManifestReader reader(data, size);
  if (!reader.skip(kCudaFqnWeightsMagicSize) ||
      !reader.read_string(manifest.so_blob_key) ||
      manifest.so_blob_key.empty()) {
    return Error::InvalidProgram;
  }

  uint32_t num_entries = 0;
  constexpr uint32_t kMaxManifestEntries = 1U << 20;
  if (!reader.read_u32(num_entries) || num_entries > kMaxManifestEntries) {
    return Error::InvalidProgram;
  }
  manifest.entries.clear();
  manifest.entries.reserve(num_entries);

  constexpr uint32_t kMaxTensorDimensions = 64;
  for (uint32_t index = 0; index < num_entries; ++index) {
    CudaFqnWeightEntry entry;
    uint32_t ndim = 0;
    if (!reader.read_string(entry.fqn) || entry.fqn.empty() ||
        !reader.read_string(entry.storage_key) || entry.storage_key.empty() ||
        !reader.read_u64(entry.storage_nbytes) ||
        !reader.read_i32(entry.dtype) ||
        !is_supported_cuda_fqn_dtype(entry.dtype) ||
        !reader.read_i32(entry.device_type) ||
        !is_supported_cuda_fqn_device_type(entry.device_type) ||
        !reader.read_i64(entry.storage_offset) || !reader.read_u32(ndim) ||
        ndim > kMaxTensorDimensions) {
      return Error::InvalidProgram;
    }

    entry.sizes.resize(ndim);
    entry.strides.resize(ndim);
    for (uint32_t dim = 0; dim < ndim; ++dim) {
      if (!reader.read_i64(entry.sizes[dim]) || entry.sizes[dim] < 0) {
        return Error::InvalidProgram;
      }
    }
    for (uint32_t dim = 0; dim < ndim; ++dim) {
      if (!reader.read_i64(entry.strides[dim]) || entry.strides[dim] < 0) {
        return Error::InvalidProgram;
      }
    }
    if (entry.storage_offset < 0) {
      return Error::InvalidProgram;
    }
    manifest.entries.push_back(std::move(entry));
  }

  return reader.empty() ? Error::Ok : Error::InvalidProgram;
}

} // namespace executorch::backends::cuda
