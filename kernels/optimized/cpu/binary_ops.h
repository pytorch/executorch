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
};

namespace internal {
inline ElementwiseOptimizedPath select_broadcast_2d_by_1d_optimized_path(
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
  return internal::select_broadcast_2d_by_1d_optimized_path(a, b);
}

} // namespace executor
} // namespace torch
