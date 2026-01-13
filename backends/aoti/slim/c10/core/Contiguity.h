/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/runtime/core/array_ref.h>

namespace executorch::backends::aoti::slim::c10 {

using ::executorch::runtime::ArrayRef;

/**
 * Compute whether a tensor with given sizes, strides, and numel is contiguous.
 *
 * A tensor is contiguous if its elements are laid out in memory in row-major
 * order, i.e., the stride of the last dimension is 1, and each preceding
 * dimension's stride equals the product of all following dimensions' sizes.
 *
 * @param sizes The sizes of each dimension
 * @param strides The strides of each dimension
 * @param numel The total number of elements
 * @return true if the tensor is contiguous, false otherwise
 */
template <typename T>
bool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {
  if (numel == 0) {
    return true;
  }

  T expected_stride = 1;
  // Iterate from last dimension to first
  for (int64_t d = static_cast<int64_t>(sizes.size()) - 1; d >= 0; d--) {
    const auto& size_d = sizes[d];
    if (size_d == 1) {
      // Size-1 dimensions don't affect contiguity
      continue;
    }

    if (strides[d] != expected_stride) {
      return false;
    }
    expected_stride *= size_d;
  }
  return true;
}

} // namespace executorch::backends::aoti::slim::c10
