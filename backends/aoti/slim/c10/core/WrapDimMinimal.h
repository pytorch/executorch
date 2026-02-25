/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <utility>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim::c10 {

namespace detail {

/// Slow path for maybe_wrap_dim when dimension needs validation.
template <typename T>
inline T maybe_wrap_dim_slow(T dim, T dim_post_expr, bool wrap_scalar) {
  ET_CHECK_MSG(
      dim_post_expr >= 0,
      "Rank cannot be negative but got %ld",
      static_cast<long>(dim_post_expr));

  if (dim_post_expr == 0) {
    ET_CHECK_MSG(
        wrap_scalar,
        "Dimension specified as %ld but tensor has no dimensions",
        static_cast<long>(dim));
    // Recursively call with dim_post_expr=1
    if (dim >= 0 && dim < 1) {
      return dim;
    } else if (dim >= -1 && dim < 0) {
      return dim + 1;
    }
    ET_CHECK_MSG(
        false,
        "Dimension out of range (expected to be in range of [-1, 0], but got %ld)",
        static_cast<long>(dim));
  }

  T min = dim_post_expr * -1;
  T max = dim_post_expr - 1;
  ET_CHECK_MSG(
      min <= dim && dim <= max,
      "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
      static_cast<long>(min),
      static_cast<long>(max),
      static_cast<long>(dim));

  // This should be unreachable if above check passes
  return dim < 0 ? dim + dim_post_expr : dim;
}

} // namespace detail

/// Wraps a dimension index to handle negative indexing.
/// For example, dim=-1 with dim_post_expr=3 returns 2.
///
/// @param dim The dimension index (may be negative).
/// @param dim_post_expr The number of dimensions.
/// @param wrap_scalar If true, allows wrapping for 0-dimensional tensors.
/// @return The wrapped dimension index (always non-negative).
template <typename T>
inline T _maybe_wrap_dim(T dim, T dim_post_expr, bool wrap_scalar = true) {
  // Inline the fast paths
  if (SLIMTENSOR_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    if (dim < 0) {
      return dim + dim_post_expr;
    }
    return dim;
  }
  // Check edge-cases out-of-line
  return detail::maybe_wrap_dim_slow<T>(
      std::move(dim), std::move(dim_post_expr), wrap_scalar);
}

/// Wraps a dimension index for int64_t.
inline int64_t
maybe_wrap_dim(int64_t dim, int64_t dim_post_expr, bool wrap_scalar = true) {
  return _maybe_wrap_dim(dim, dim_post_expr, wrap_scalar);
}

/// Wraps a dimension index for size_t.
inline int64_t
maybe_wrap_dim(int64_t dim, size_t dim_post_expr, bool wrap_scalar = true) {
  return _maybe_wrap_dim(dim, static_cast<int64_t>(dim_post_expr), wrap_scalar);
}

} // namespace executorch::backends::aoti::slim::c10
