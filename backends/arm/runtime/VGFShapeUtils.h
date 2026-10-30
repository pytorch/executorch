/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace executorch {
namespace backends {
namespace vgf {
namespace detail {

/*
 * Validate a tensor shape and calculate its element count.
 *
 * Shapes decoded from VGF are untrusted input. In particular, multiplying
 * dimensions must not wrap either the int64_t domain used by tensor metadata
 * or size_t used for host allocation sizes.
 */
inline bool checked_element_count_from_shape(
    const std::vector<int64_t>& shape,
    size_t* element_count) {
  if (element_count == nullptr) {
    return false;
  }

  if (shape.empty()) {
    *element_count = 1;
    return true;
  }

  constexpr uint64_t kMaxInt64 =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  const uint64_t max_size_t =
      static_cast<uint64_t>(std::numeric_limits<size_t>::max());
  const uint64_t max_count = max_size_t < kMaxInt64 ? max_size_t : kMaxInt64;

  uint64_t count = 1;

  for (const int64_t dim : shape) {
    if (dim <= 0) {
      return false;
    }

    const uint64_t unsigned_dim = static_cast<uint64_t>(dim);

    if (count > max_count / unsigned_dim) {
      return false;
    }

    count *= unsigned_dim;
  }

  *element_count = static_cast<size_t>(count);
  return true;
}

/*
 * Calculate the logical byte size of a tensor without overflowing size_t.
 */
inline bool checked_logical_byte_size_from_shape(
    const std::vector<int64_t>& shape,
    size_t element_size,
    size_t* logical_byte_size) {
  if (logical_byte_size == nullptr || element_size == 0) {
    return false;
  }

  size_t element_count = 0;
  if (!checked_element_count_from_shape(shape, &element_count)) {
    return false;
  }

  if (element_count > std::numeric_limits<size_t>::max() / element_size) {
    return false;
  }

  *logical_byte_size = element_count * element_size;
  return true;
}

/*
 * Return the supplied stride, or generate a checked contiguous stride when
 * no explicit stride was encoded in the VGF.
 *
 * The shape is validated even when an explicit stride exists so malformed
 * dimensions cannot bypass shape validation.
 */
inline bool normalize_stride_checked(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    std::vector<int64_t>* normalized_stride) {
  if (normalized_stride == nullptr) {
    return false;
  }

  size_t element_count = 0;
  if (!checked_element_count_from_shape(shape, &element_count)) {
    return false;
  }

  (void)element_count;

  if (!stride.empty()) {
    // VkTensorDescriptionARM::pStrides must have one entry per dimension.
    if (stride.size() != shape.size()) {
      return false;
    }
    *normalized_stride = stride;
    return true;
  }

  normalized_stride->assign(shape.size(), 1);

  int64_t running = 1;
  for (size_t idx = shape.size(); idx > 0; --idx) {
    const int64_t dim = shape[idx - 1];

    /*
     * checked_element_count_from_shape() has already verified dim > 0,
     * but keep the condition local to the multiplication as defense in depth.
     */
    if (dim <= 0) {
      return false;
    }

    (*normalized_stride)[idx - 1] = running;

    if (running > std::numeric_limits<int64_t>::max() / dim) {
      return false;
    }

    running *= dim;
  }

  return true;
}

} // namespace detail
} // namespace vgf
} // namespace backends
} // namespace executorch
