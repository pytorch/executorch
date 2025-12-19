/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/slim/c10/util/irange.h>
#include <executorch/runtime/core/array_ref.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace executorch::backends::aoti::slim::c10 {

using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::IntArrayRef;

template <typename T>
bool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {
  if (numel == 0) {
    return true;
  }

  T expected_stride = 1;
  // NB: make sure we do signed arithmetic
  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {
    const auto& size_d = sizes[d];
    if (size_d == 1) {
      continue;
    }

    if (strides[d] != expected_stride) {
      return false;
    }
    expected_stride *= size_d;
  }
  return true;
}

// This function will return True if the tensor is contiguous, and False if the
// its not or if we can't determine if it is contiguous due to unbacked symbols
// (it could be either in that case based on the actual runtime data).
template <typename T>
bool definitely_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {
  if (numel == 0) {
    return true;
  }

  T expected_stride = 1;
  // NB: make sure we do signed arithmetic
  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {
    const auto& size_d = sizes[d];
    if (size_d == 1) {
      continue;
    }

    if (strides[d] != expected_stride) {
      return false;
    }
    expected_stride *= size_d;
  }
  return true;
}

template <typename T>
bool _compute_channels_last_contiguous_2d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {
    case 4: {
      T expected = 1;
      for (auto& d : {1, 3, 2, 0}) {
        const auto& size_d = sizes[d];
        if (size_d != 1) {
          if (strides[d] != expected) {
            return false;
          }
          expected *= size_d;
        }
      }
      return true;
    }
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 3:
      // TODO dim == 3 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

template <typename T>
bool _compute_channels_last_contiguous_3d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {
    case 5: {
      T expected = 1;
      for (auto& d : {1, 4, 3, 2, 0}) {
        const auto& size_d = sizes[d];
        if (size_d != 1) {
          if (strides[d] != expected) {
            return false;
          }
          expected *= size_d;
        }
      }
      return true;
    }
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

template <typename T>
bool _compute_non_overlapping_and_dense(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  auto dim = sizes.size();
  if (dim == 1) {
    return sizes[0] < 2 || strides[0] == 1;
  }
  std::vector<int64_t> perm(dim);
  for (const auto i : irange(dim)) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    if (sizes[a] < 2) {
      return false;
    } else if (sizes[b] < 2) {
      return true;
    }
    return strides[a] < strides[b];
  });
  T require_stride = 1;
  for (const auto i : irange(dim)) {
    const auto& size_perm_i = sizes[perm[i]];
    if (size_perm_i < 2) {
      return true;
    }
    if (strides[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= size_perm_i;
  }
  return true;
}

} // namespace executorch::backends::aoti::slim::c10
