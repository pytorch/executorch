#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include <executorch/backends/aoti/slim/c10/util/accumulate.h>
#include <executorch/backends/aoti/slim/c10/util/irange.h>
#include <executorch/backends/aoti/slim/c10/util/safe_numerics.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>

namespace executorch::backends::aoti::slim {
#ifndef STANDALONE_MOBILE
inline constexpr uint64_t storage_max() {
  // int64_t and size_t are used somewhat inconsistently throughout ATen.
  // To be safe, storage size calculations must fit in both types.
  constexpr auto int64_max =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  constexpr auto size_max =
      static_cast<uint64_t>(std::numeric_limits<size_t>::max());
  return std::min(int64_max, size_max);
}

/**
 * Compute the number of elements based on the sizes of a
 * tensor. Catches integer overflow that may occur when a tensor
 * using a sparse layout has multiple dimensions with large sizes.
 */
inline int64_t safe_compute_numel(IntArrayRef sizes) {
  uint64_t n = 1;
  bool overflowed =
      executorch::backends::aoti::slim::c10::safe_multiplies_u64(sizes, &n);
  overflowed |= (n > storage_max());
  ET_CHECK_MSG(!overflowed, "numel: integer multiplication overflow");
  return static_cast<int64_t>(n);
}

inline std::vector<int64_t> safe_compute_contiguous_strides(IntArrayRef sizes) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  std::vector<int64_t> strides(ndim);
  if (ndim > 0) {
    uint64_t stride = 1;
    bool overflowed = false;
    for (int64_t i = ndim - 1; i >= 0; i--) {
      strides[i] = static_cast<int64_t>(stride);
      if (sizes[i] != 0) {
        uint64_t new_stride = 0;
        overflowed |= c10::mul_overflows(
            stride, static_cast<uint64_t>(sizes[i]), &new_stride);
        stride = new_stride;
      }
    }
    ET_CHECK_MSG(
        !overflowed, "contiguous_strides: stride multiplication overflow");
  }
  return strides;
}
#endif // STANDALONE_MOBILE

inline int64_t compute_numel(IntArrayRef sizes) {
#ifndef STANDALONE_MOBILE
  // Use overflow checks if supported by the compiler
  return safe_compute_numel(sizes);
#else
  return executorch::backends::aoti::slim::c10::multiply_integers(sizes);
#endif
}

// named computeStorageNbytesContiguous in c10
inline size_t compute_storage_nbytes_contiguous(
    IntArrayRef sizes,
    size_t itemsize_bytes,
    size_t storage_offset) {
// Ignore overflow checks on mobile
#ifndef STANDALONE_MOBILE
  uint64_t size = 1;
  bool overflowed =
      executorch::backends::aoti::slim::c10::safe_multiplies_u64(sizes, &size);
  overflowed |= executorch::backends::aoti::slim::c10::add_overflows(
      size, storage_offset, &size);
  overflowed |= executorch::backends::aoti::slim::c10::mul_overflows(
      size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  ET_CHECK_MSG(!overflowed, "Storage size calculation overflowed");
  return static_cast<size_t>(size);
#else
  const auto numel = multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
#endif
}

// named computeStorageNbytes in c10
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

// Ignore overflow checks on mobile
#ifndef STANDALONE_MOBILE
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = storage_offset + 1;
  bool overflowed = false;
  for (const auto i :
       executorch::backends::aoti::slim::c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    uint64_t strided_size = 0;
    overflowed |= executorch::backends::aoti::slim::c10::mul_overflows(
        strides[i], sizes[i] - 1, &strided_size);
    overflowed |= executorch::backends::aoti::slim::c10::add_overflows(
        size, strided_size, &size);
  }
  overflowed |= executorch::backends::aoti::slim::c10::mul_overflows(
      size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  ET_CHECK_MSG(!overflowed, "Storage size calculation overflowed");
  return static_cast<size_t>(size);
#else
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = 1;
  for (const auto i :
       executorch::backends::aoti::slim::c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
#endif
}

inline std::vector<int64_t> compute_contiguous_strides(IntArrayRef sizes) {
#ifndef STANDALONE_MOBILE
  return safe_compute_contiguous_strides(sizes);
#else
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
#endif
}

// calculates the final concrete shape by also filling in at most one '-1'
// dimension.
inline std::vector<int64_t> infer_size(IntArrayRef shape, int64_t numel) {
  int64_t new_size = 1;
  std::optional<int64_t> infer_dim;
  std::vector<int64_t> result_shape;
  result_shape.reserve(shape.size());

  size_t ndim = shape.size();
  bool overflowed = false;
  for (size_t dim = 0; dim < ndim; dim++) {
    if (shape[dim] == -1) {
      ET_CHECK_MSG(
          !infer_dim.has_value(), "only one dimension can be inferred");
      infer_dim = dim;
      result_shape.push_back(-1); // placeholder
    } else {
      ET_CHECK_MSG(
          shape[dim] >= 0,
          "invalid shape dimension %ld",
          static_cast<long>(shape[dim]));
      overflowed |= executorch::backends::aoti::slim::c10::mul_overflows(
          new_size, shape[dim], &new_size);
      result_shape.push_back(shape[dim]);
    }
  }
  ET_CHECK_MSG(!overflowed, "shape calculation overflowed");

  if (infer_dim.has_value()) {
    ET_CHECK_MSG(
        new_size != 0,
        "cannot reshape tensor of 0 elements into shape with -1");
    ET_CHECK_MSG(
        numel % new_size == 0,
        "shape is invalid for input size %ld",
        static_cast<long>(numel));
    result_shape[*infer_dim] = numel / new_size;
  } else {
    ET_CHECK_MSG(
        numel == new_size,
        "shape is invalid for input of size %ld",
        static_cast<long>(numel));
  }
  return result_shape;
}

// it determines if a reshape is possible as a view.
// If so, it returns the new strides
// If not, it returns an empty optional
inline std::optional<std::vector<int64_t>> compute_stride(
    IntArrayRef old_sizes,
    IntArrayRef old_strides,
    IntArrayRef new_sizes) {
  if (old_sizes.empty()) {
    return std::vector<int64_t>(new_sizes.size(), 1);
  }

  // NOTE: stride is arbitrary in the numel() == 0 case;
  // to match NumPy behavior we copy the strides if the size matches, otherwise
  // we use the stride as if it were computed via resize.
  // This could perhaps be combined with the below code, but the complexity
  // didn't seem worth it.
  size_t numel = compute_numel(old_sizes);
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
  bool overflowed = false;
  for (int64_t tensor_d = static_cast<int64_t>(old_sizes.size()) - 1;
       tensor_d >= 0;
       tensor_d--) {
    // TODO: ask if this could lead to overflow by any chance?
    // even if so, overflow is not handled in the aten implementation
    overflowed |= executorch::backends::aoti::slim::c10::mul_overflows(
        tensor_numel, old_sizes[tensor_d], &tensor_numel);

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
  ET_CHECK_MSG(!overflowed, "overflowed while computing strides");

  if (view_d != -1) {
    return std::nullopt; // not viewable
  }
  return new_strides;
}

} // namespace executorch::backends::aoti::slim
