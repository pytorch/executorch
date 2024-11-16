/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace internal {
// NOTE: we bake ArrayRef iterators being pointers into the return
// type here because we assume that iterators are portable across
// ArrayRef copies.
inline const Tensor::SizesType* arrayref_begin_ignoring_leading_1s(
    ArrayRef<Tensor::SizesType> arr) {
  return std::find_if(
      arr.begin(), arr.end(), [](Tensor::SizesType x) { return x != 1; });
}

inline bool sizes_match_ignoring_leading_1s(
    ArrayRef<Tensor::SizesType> lhs,
    ArrayRef<Tensor::SizesType> rhs) {
  auto lhs_begin = arrayref_begin_ignoring_leading_1s(lhs);
  auto lhs_end = lhs.end();

  auto rhs_begin = arrayref_begin_ignoring_leading_1s(rhs);
  auto rhs_end = rhs.end();

  return ((lhs_end - lhs_begin) == (rhs_end - rhs_begin)) &&
      std::equal(lhs_begin, lhs_end, rhs_begin);
}
} // namespace internal

enum class ElementwiseOptimizedPath {
  kNone,
  kTreatAs1d,
  kBroadcast2dBy1d,
  kBroadcast2dBy1dReverseArguments,
  kBroadcastNdByNd,
  kBroadcastNdByNdReverseArguments,
  kBroadcastLastDim,
  kBroadcastLastDimReverseArguments,
};

namespace internal {

/*
  Given two tensors, this function returns the broadcast dim if it exists.
  Returns 0 if no broadcast dim is found.
  Else negative index is used to indicate broadcast dim
  e.g. if size = [a, b, c, 1, e, f] then broadcast dim is -3

  This path aims to handle broadcast of the following form
  A = [a1, a2,., 1, .., an]
  B = [b1, b2,., bm, .., bn]
  OR
  A = [a1, a2,., am, .., an]
  B = [b1, b2,., 1, .., bn]
  Note that this way of determining broadcast dim also works
  when broadcast dim is the last dim.
*/
int32_t inline get_broadcast_dim(const Tensor& lhs, const Tensor& rhs) {
  auto lhs_begin = arrayref_begin_ignoring_leading_1s(lhs.sizes());
  auto lhs_end = lhs.sizes().end();

  auto rhs_begin = arrayref_begin_ignoring_leading_1s(rhs.sizes());
  auto rhs_end = rhs.sizes().end();

  const auto lhs_size = lhs_end - lhs_begin;
  const auto rhs_size = rhs_end - rhs_begin;

  // Following example is not handled at the moment
  // [1, 3, 4, 5]
  // [2, 3, 4, 5]
  if (lhs_size != rhs_size) {
    return 0;
  }

  int32_t broadcast_dim = 0;
  // Check
  // 1. if any dim value is 1 (it constitutes a broadcast dim)
  // 2. If more than one dim value is 1 (we cannot handle)
  // 3. If non-1 dim values are equal
  lhs_end--;
  rhs_end--;
  while (lhs_end != lhs_begin) {
    if (*lhs_end == 1 || *rhs_end == 1) {
      // If more than one broadcast dim is found, return 0.
      if (broadcast_dim != 0) {
        return 0;
      }
      // negative index is used
      broadcast_dim = lhs_end - lhs.sizes().end();
    } else if (*lhs_end != *rhs_end) {
      // If non-1 dim values are not equal, return 0.
      return 0;
    }
    lhs_end--;
    rhs_end--;
  }
  return broadcast_dim;
}

inline ElementwiseOptimizedPath select_broadcast_optimized_path(
    const Tensor& lhs,
    const Tensor& rhs) {
  auto lhs_begin = arrayref_begin_ignoring_leading_1s(lhs.sizes());
  auto lhs_end = lhs.sizes().end();

  auto rhs_begin = arrayref_begin_ignoring_leading_1s(rhs.sizes());
  auto rhs_end = rhs.sizes().end();

  const auto lhs_size = lhs_end - lhs_begin;
  const auto rhs_size = rhs_end - rhs_begin;
  if (lhs_size == 2 && rhs_size == 1 && lhs_begin[1] == rhs_begin[0]) {
    return ElementwiseOptimizedPath::kBroadcast2dBy1d;
  }

  if (lhs_size == 1 && rhs_size == 2 && rhs_begin[1] == lhs_begin[0]) {
    return ElementwiseOptimizedPath::kBroadcast2dBy1dReverseArguments;
  }

  int32_t broadcast_dim = get_broadcast_dim(lhs, rhs);
  // Right now we dont handle last dim broadcast
  if (broadcast_dim < -1) {
    if (std::count_if(rhs_begin, rhs_end, [](Tensor::SizesType x) {
          return x == 1;
        }) == 1) {
      return ElementwiseOptimizedPath::kBroadcastNdByNd;
    } else {
      return ElementwiseOptimizedPath::kBroadcastNdByNdReverseArguments;
    }
  } else if (broadcast_dim == -1) {
    if (std::count_if(lhs_begin, lhs_end, [](Tensor::SizesType x) {
          return x == 1;
        }) == 1) {
      return ElementwiseOptimizedPath::kBroadcastLastDimReverseArguments;
    } else {
      return ElementwiseOptimizedPath::kBroadcastLastDim;
    }
  }
  return ElementwiseOptimizedPath::kNone;
}
} // namespace internal

ElementwiseOptimizedPath inline select_optimized_path(
    const Tensor& a,
    const Tensor& b,
    const Tensor& out) {
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  if (a_type != b_type || a_type != out_type || a_type == ScalarType::Half ||
      a_type == ScalarType::BFloat16) {
    return ElementwiseOptimizedPath::kNone;
  }
  if (a.sizes().equals(b.sizes()) ||
      (a.numel() == b.numel() &&
       (a.numel() == out.numel() ||
        internal::sizes_match_ignoring_leading_1s(a.sizes(), b.sizes())))) {
    return ElementwiseOptimizedPath::kTreatAs1d;
  }
  return internal::select_broadcast_optimized_path(a, b);
}

std::array<int32_t, 3> inline get_normalized_tensor_size(
    const Tensor& a,
    const int32_t broadcast_dim) {
  ET_CHECK_MSG(
      a.dim() > broadcast_dim,
      "Size of tensor: %zd, must be larger than broadcast_dim: %d",
      a.dim(),
      broadcast_dim);
  std::array<int32_t, 3> normalized_tensor_size;
  normalized_tensor_size[0] = 1;
  normalized_tensor_size[1] = a.size(broadcast_dim);
  normalized_tensor_size[2] = 1;
  for (size_t i = 0; i < broadcast_dim; i++) {
    normalized_tensor_size[0] *= a.size(i);
  }
  for (size_t i = broadcast_dim + 1; i < a.dim(); i++) {
    normalized_tensor_size[2] *= a.size(i);
  }
  return normalized_tensor_size;
}

} // namespace executor
} // namespace torch
