#pragma once

#include <cstdint>
#include <utility>

#include <executorch/backends/aoti/slim/c10/macros/Macros.h>

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
  STANDALONE_CHECK(
      dim_post_expr >= 0, "Rank cannot be negative but got ", dim_post_expr);

  if (dim_post_expr == 0) {
    STANDALONE_CHECK(
        wrap_scalar,
        "Dimension specified as ",
        dim,
        " but tensor has no dimensions");
    return executorch::backends::aoti::slim::c10::maybe_wrap_dim(
        std::move(dim),
        /*dim_post_expr=*/1,
        /*wrap_scalar=*/false);
  }

  T min = dim_post_expr * -1;
  T max = dim_post_expr - 1;
  STANDALONE_CHECK(
      min <= dim && dim <= max,
      "Dimension out of range (expected to be in range of [",
      min,
      ", ",
      max,
      "], but got ",
      dim,
      ")");

  STANDALONE_INTERNAL_ASSERT(
      false, "should never reach here as dim should be out-of-bounds");
}
} // namespace detail
} // namespace executorch::backends::aoti::slim::c10
