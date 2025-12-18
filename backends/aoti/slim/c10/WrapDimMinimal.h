/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/macros/Macros.h>
#include <executorch/runtime/platform/assert.h>

#include <cstdint>
#include <utility>

// Different from the original implementation in c10, we don't need
// to support SymInt here.
namespace c10 {
namespace detail {
template <typename T>
T maybe_wrap_dim_slow(T dim, T dim_post_expr, bool wrap_scalar);
}

template <typename T>
T _maybe_wrap_dim(T dim, T dim_post_expr, bool wrap_scalar = true) {
  // Inline the fast paths
  if (C10_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    // For SymInts, we want an explicit control flow to trigger a guard, so we
    // may as well branch too.
    if (dim < 0) {
      return dim + dim_post_expr;
    }
    return dim;
  }
  // Check edge-cases out-of-line (wrapping scalars and out-of-bounds errors)
  return c10::detail::maybe_wrap_dim_slow<T>(
      std::move(dim), std::move(dim_post_expr), wrap_scalar);
}

inline int64_t
maybe_wrap_dim(int64_t dim, int64_t dim_post_expr, bool wrap_scalar = true) {
  return _maybe_wrap_dim(dim, dim_post_expr, wrap_scalar);
}

namespace detail {
// This template can only be specialized at int64_t and c10::SymInt;
// you'll get linker errors otherwise
template <typename T>
T maybe_wrap_dim_slow(T dim, T dim_post_expr, bool wrap_scalar) {
  ET_CHECK_MSG(
      dim_post_expr >= 0,
      "Rank cannot be negative but got %" PRId64,
      static_cast<int64_t>(dim_post_expr));

  if (dim_post_expr == 0) {
    ET_CHECK_MSG(
        wrap_scalar,
        "Dimension specified as %" PRId64 " but tensor has no dimensions",
        static_cast<int64_t>(dim));
    return c10::maybe_wrap_dim(
        std::move(dim), /*dim_post_expr=*/1, /*wrap_scalar=*/false);
  }

  T min = dim_post_expr * -1;
  T max = dim_post_expr - 1;
  ET_CHECK_MSG(
      min <= dim && dim <= max,
      "Dimension out of range (expected to be in range of [%" PRId64
      ", %" PRId64 "], but got %" PRId64 ")",
      static_cast<int64_t>(min),
      static_cast<int64_t>(max),
      static_cast<int64_t>(dim));

  ET_DCHECK_MSG(
      false, "should never reach here as dim should be out-of-bounds");
  return dim; // unreachable, but needed to suppress compiler warnings
}
} // namespace detail
} // namespace c10
