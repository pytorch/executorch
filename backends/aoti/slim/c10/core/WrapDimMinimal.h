#pragma once

#include <cstdint>
#include <utility>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/runtime/platform/assert.h>

// Different from the original implementation in c10, we don't need
// to support SymInt here.
namespace executorch::backends::aoti::slim::c10 {
namespace detail {
template <typename T>
T maybe_wrap_dim_slow(T dim, T dim_post_expr, bool wrap_scalar);
}

template <typename T>
T _maybe_wrap_dim(T dim, T dim_post_expr, bool wrap_scalar = true) {
  // Inline the fast paths
  if (STANDALONE_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    // For SymInts, we want an explicit control flow to trigger a guard, so we
    // may as well branch too.
    if (dim < 0) {
      return dim + dim_post_expr;
    }
    return dim;
  }
  // Check edge-cases out-of-line (wrapping scalars and out-of-bounds errors)
  return executorch::backends::aoti::slim::c10::detail::maybe_wrap_dim_slow<T>(
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
      "Rank cannot be negative but got %ld",
      static_cast<long>(dim_post_expr));

  if (dim_post_expr == 0) {
    ET_CHECK_MSG(
        wrap_scalar,
        "Dimension specified as %ld but tensor has no dimensions",
        static_cast<long>(dim));
    return executorch::backends::aoti::slim::c10::maybe_wrap_dim(
        std::move(dim),
        /*dim_post_expr=*/1,
        /*wrap_scalar=*/false);
  }

  T min = dim_post_expr * -1;
  T max = dim_post_expr - 1;
  ET_CHECK_MSG(
      min <= dim && dim <= max,
      "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
      static_cast<long>(min),
      static_cast<long>(max),
      static_cast<long>(dim));

  ET_DCHECK_MSG(
      false, "should never reach here as dim should be out-of-bounds");
}
} // namespace detail
} // namespace executorch::backends::aoti::slim::c10
