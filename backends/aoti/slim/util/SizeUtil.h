/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim {

/// Computes the number of elements (numel) from the sizes array.
/// @param sizes Array of dimension sizes.
/// @return The total number of elements (product of all sizes).
inline int64_t compute_numel(IntArrayRef sizes) {
  int64_t n = 1;
  for (auto s : sizes) {
    n *= s;
  }
  return n;
}

/// Computes the storage size in bytes for a contiguous tensor.
/// @param sizes Array of dimension sizes.
/// @param itemsize_bytes Size of each element in bytes.
/// @param storage_offset Offset into the storage in elements.
/// @return The required storage size in bytes.
inline size_t compute_storage_nbytes_contiguous(
    IntArrayRef sizes,
    size_t itemsize_bytes,
    size_t storage_offset) {
  const auto numel = compute_numel(sizes);
  return itemsize_bytes * (storage_offset + numel);
}

/// Computes the storage size in bytes for a strided tensor.
/// @param sizes Array of dimension sizes.
/// @param strides Array of dimension strides.
/// @param itemsize_bytes Size of each element in bytes.
/// @param storage_offset Offset into the storage in elements.
/// @return The required storage size in bytes.
inline size_t compute_storage_nbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes,
    size_t storage_offset) {
  ET_CHECK_MSG(
      sizes.size() == strides.size(),
      "dimensionality of sizes (%zu) must match dimensionality of strides (%zu)",
      sizes.size(),
      strides.size());

  // Size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride.
  size_t size = 1;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
}

/// Computes contiguous strides from sizes.
/// @param sizes Array of dimension sizes.
/// @return Vector of strides for a contiguous tensor.
inline std::vector<int64_t> compute_contiguous_strides(IntArrayRef sizes) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  std::vector<int64_t> strides(ndim);
  if (ndim > 0) {
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; i--) {
      strides[i] = stride;
      if (sizes[i] != 0) {
        stride *= sizes[i];
      }
    }
  }
  return strides;
}

} // namespace executorch::backends::aoti::slim
