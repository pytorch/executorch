/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <executorch/backends/aoti/slim/util/array_ref_util.h>
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

/// Infers the final concrete shape by filling in at most one '-1' dimension.
/// @param shape The proposed shape, may contain one -1 for inference.
/// @param numel The total number of elements in the tensor.
/// @return Vector with the final shape (no -1 entries).
inline std::vector<int64_t> infer_size(IntArrayRef shape, int64_t numel) {
  int64_t new_size = 1;
  int64_t infer_dim = -1;
  std::vector<int64_t> result_shape;
  result_shape.reserve(shape.size());

  for (size_t dim = 0; dim < shape.size(); dim++) {
    if (shape[dim] == -1) {
      ET_CHECK_MSG(infer_dim == -1, "only one dimension can be inferred");
      infer_dim = static_cast<int64_t>(dim);
      result_shape.push_back(-1); // placeholder
    } else {
      ET_CHECK_MSG(
          shape[dim] >= 0,
          "invalid shape dimension %ld",
          static_cast<long>(shape[dim]));
      new_size *= shape[dim];
      result_shape.push_back(shape[dim]);
    }
  }

  if (infer_dim != -1) {
    ET_CHECK_MSG(
        new_size != 0,
        "cannot reshape tensor of 0 elements into shape with -1");
    ET_CHECK_MSG(
        numel % new_size == 0,
        "shape is invalid for input size %ld",
        static_cast<long>(numel));
    result_shape[static_cast<size_t>(infer_dim)] = numel / new_size;
  } else {
    ET_CHECK_MSG(
        numel == new_size,
        "shape is invalid for input of size %ld",
        static_cast<long>(numel));
  }
  return result_shape;
}

/// Determines if a reshape is possible as a view without copying.
/// If so, returns the new strides; otherwise returns an empty optional.
/// @param old_sizes Current tensor sizes.
/// @param old_strides Current tensor strides.
/// @param new_sizes Target tensor sizes.
/// @return Strides for the view, or nullopt if copy is required.
inline std::optional<std::vector<int64_t>> compute_stride(
    IntArrayRef old_sizes,
    IntArrayRef old_strides,
    IntArrayRef new_sizes) {
  if (old_sizes.empty()) {
    return std::vector<int64_t>(new_sizes.size(), 1);
  }

  // Handle numel == 0 case
  size_t numel = static_cast<size_t>(compute_numel(old_sizes));
  if (numel == 0 && old_sizes == new_sizes) {
    return toVec(old_strides);
  }

  int64_t new_sizes_len = static_cast<int64_t>(new_sizes.size());
  std::vector<int64_t> new_strides(new_sizes_len);
  if (numel == 0) {
    for (int64_t view_d = new_sizes_len - 1; view_d >= 0; view_d--) {
      if (view_d == new_sizes_len - 1) {
        new_strides[view_d] = 1;
      } else {
        new_strides[view_d] = std::max<int64_t>(new_sizes[view_d + 1], 1) *
            new_strides[view_d + 1];
      }
    }
    return new_strides;
  }

  int64_t view_d = new_sizes_len - 1;
  int64_t chunk_base_stride = old_strides.back();
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;

  for (int64_t tensor_d = static_cast<int64_t>(old_sizes.size()) - 1;
       tensor_d >= 0;
       tensor_d--) {
    tensor_numel *= old_sizes[tensor_d];

    bool is_chunk_end = (tensor_d == 0) ||
        (old_sizes[tensor_d - 1] != 1 &&
         old_strides[tensor_d - 1] != tensor_numel * chunk_base_stride);

    if (is_chunk_end) {
      while (view_d >= 0 &&
             (view_numel < tensor_numel || new_sizes[view_d] == 1)) {
        new_strides[view_d] = view_numel * chunk_base_stride;
        view_numel *= new_sizes[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) {
        return std::nullopt; // Not viewable
      }
      if (tensor_d > 0) {
        chunk_base_stride = old_strides[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }

  if (view_d != -1) {
    return std::nullopt; // Not viewable
  }
  return new_strides;
}

} // namespace executorch::backends::aoti::slim
