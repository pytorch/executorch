/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/optimized/utils/math_utils.h>
#include <executorch/kernels/optimized/utils/unroll.h>

#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#include <array>
#include <vector>

namespace executorch {
namespace cpublas {

template <typename scalar_t, typename opmath_t>
void scale_(int64_t m, int64_t n, opmath_t alpha, scalar_t* a, int64_t lda) {
  if (alpha == opmath_t(1)) {
    return; // identity
  }

  if (alpha == opmath_t(0)) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < m; ++i) {
        a[j * lda + i] = scalar_t(0);
      }
    }
    return;
  }

  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      a[j * lda + i] *= alpha;
    }
  }
}

template <typename Func>
auto sum(int64_t N, Func f) {
  constexpr int ilp_factor = 4;
  using acc_t = decltype(f(0));

  // Calculate independent partial sums then add together at the end
  std::array<acc_t, ilp_factor> partial_sums{};

  size_t i = 0;
  for (; i + ilp_factor <= N; i += ilp_factor) {
    utils::ForcedUnroll<ilp_factor>{}(
        [&i, &f, &partial_sums](int k) { partial_sums[k] += f(i + k); });
  }
  for (; i < N; ++i) {
    partial_sums[0] += f(i);
  }
  for (int k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }
  return partial_sums[0];
}

template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b)
  for (size_t l = 0; l < k; ++l) {
    for (size_t j = 0; j < n; ++j) {
      opmath_t val = b[l + j * ldb] * alpha;
      int64_t i_m = m / 4;
      for (int64_t i_i = 0; i_i < i_m; ++i_i) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++) {
        c[j * ldc + i] += a[i + l * lda] * val;
      }
    }
  }
}

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
// out_t defaults to scalar_t; pass a wider out_t (e.g. float) to accumulate a
// reduced-precision matmul into a full-precision output.
template <typename scalar_t, typename opmath_t, typename out_t = scalar_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    out_t* c,
    int64_t ldc) {
  // c += alpha * (a @ b)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(b[j * ldb + l]);
      });
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = static_cast<out_t>(alpha * dot);
      } else {
        c[j * ldc + i] = static_cast<out_t>(
            beta * static_cast<opmath_t>(c[j * ldc + i]) + alpha * dot);
      }
    }
  }
}

namespace internal {
float bf16_dot_with_fp32_arith(
    const torch::executor::BFloat16* vec1,
    const torch::executor::BFloat16* vec2,
    int64_t len);
// Four dots of vec1 against vec2, vec2 + stride2, ... into out[0..3].
void bf16_dot4_with_fp32_arith(
    const torch::executor::BFloat16* vec1,
    const torch::executor::BFloat16* vec2,
    int64_t stride2,
    int64_t len,
    float* out);
void bf16_gemv_notrans_with_fp32_arith(
    int64_t m,
    int64_t k,
    float alpha,
    const torch::executor::BFloat16* a,
    int64_t lda,
    const torch::executor::BFloat16* b,
    float beta,
    float* c);
void bf16_fp32_gemv_notrans_with_fp32_arith(
    int64_t m,
    int64_t k,
    float alpha,
    const torch::executor::BFloat16* a,
    int64_t lda,
    const float* b,
    float beta,
    float* c);
void bf16_gemv_transa_with_fp32_arith(
    int64_t m,
    int64_t k,
    float alpha,
    const torch::executor::BFloat16* a,
    int64_t lda,
    const torch::executor::BFloat16* b,
    float beta,
    float* c);
} // namespace internal

// Used by custom SDPA's attn@V. Serial on purpose: SDPA already parallelizes
// its outer head loop, and executorch's threadpool deadlocks on a nested
// parallel_for.
// clang-format off
template <>
inline typename std::enable_if<
    !std::is_same<torch::executor::BFloat16, float>::value,
    void>::type
gemm_notrans_<torch::executor::BFloat16, float, float>(
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const torch::executor::BFloat16 *a, int64_t lda,
    const torch::executor::BFloat16 *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc) {
  if (n == 1) {
    internal::bf16_gemv_notrans_with_fp32_arith(
        m, k, alpha, a, lda, b, beta, c);
    return;
  }

  // bf16_dot_with_fp32_arith needs a contiguous k-vector, but a is strided by
  // lda in k, so the dot path must gather each row first. On aarch64 the
  // gather-free form wins for small n; x86 uses the dot path from n == 2.
  // The crossover is empirical and only visible with runtime-valued
  // dimensions: constant-folded ones let the gather-free loop unroll.
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE)
  constexpr int64_t kMinColsForGather = 32;
#else
  constexpr int64_t kMinColsForGather = 2;
#endif
  const bool use_dot = n >= kMinColsForGather;

  if (!use_dot) {
    for (int64_t j = 0; j < n; ++j) {
      float *c_col = c + j * ldc;
      if (beta == 0) {
        for (int64_t i = 0; i < m; ++i) {
          c_col[i] = 0.0f;
        }
      } else if (beta != 1.0f) {
        for (int64_t i = 0; i < m; ++i) {
          c_col[i] *= beta;
        }
      }
    }
    // l outermost so a's column is loaded once and reused across c's columns.
    for (int64_t l = 0; l < k; ++l) {
      const torch::executor::BFloat16 *a_col = a + l * lda;
      for (int64_t j = 0; j < n; ++j) {
        const float b_val = static_cast<float>(b[l + j * ldb]) * alpha;
        float *c_col = c + j * ldc;
        // Unrolled for the same reason the fp32 specialization above is: the
        // bf16->fp32 conversion does not vectorize from the rolled form.
        int64_t i = 0;
        for (; i + 4 <= m; i += 4) {
          c_col[i + 0] += static_cast<float>(a_col[i + 0]) * b_val;
          c_col[i + 1] += static_cast<float>(a_col[i + 1]) * b_val;
          c_col[i + 2] += static_cast<float>(a_col[i + 2]) * b_val;
          c_col[i + 3] += static_cast<float>(a_col[i + 3]) * b_val;
        }
        for (; i < m; ++i) {
          c_col[i] += static_cast<float>(a_col[i]) * b_val;
        }
      }
    }
    return;
  }

  // This path runs once per SDPA tile. Reuse storage across calls so gathering
  // a row does not allocate in the tile loop.
  /* library-local */ thread_local std::vector<torch::executor::BFloat16> a_row;
  a_row.resize(k);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t l = 0; l < k; ++l) {
      a_row[l] = a[l * lda + i];
    }
    const torch::executor::BFloat16 *b_ = b;
    for (int64_t j = 0; j < n; ++j) {
      const float dot = internal::bf16_dot_with_fp32_arith(a_row.data(), b_, k);
      b_ += ldb;
      if (beta == 0) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}
// clang-format on

// clang-format off
template <typename scalar_t, typename opmath_t, typename out_t = scalar_t>
void gemm_transa_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    out_t *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  const scalar_t *a_ = a;
  for (size_t i = 0; i < m; ++i) {
    const scalar_t *b_ = b;
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a_[l]) * static_cast<opmath_t>(b_[l]);
      });
      b_ += ldb;
      if (beta == opmath_t(0)) {
        c[j*ldc+i] = static_cast<out_t>(alpha*dot);
      } else {
        c[j*ldc+i] = static_cast<out_t>(beta*static_cast<opmath_t>(c[j*ldc+i])+alpha*dot);
      }
    }
    a_ += lda;
  }
}

template <>
inline void gemm_transa_<torch::executor::BFloat16, torch::executor::BFloat16, torch::executor::BFloat16>(
    int64_t m, int64_t n, int64_t k,
    torch::executor::BFloat16 alpha,
    const torch::executor::BFloat16 *a, int64_t lda,
    const torch::executor::BFloat16 *b, int64_t ldb,
    torch::executor::BFloat16 beta,
    torch::executor::BFloat16 *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  if (alpha == 1 && beta == 0) {
    executorch::extension::parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
      const auto *a_ = a + begin * lda;
      for (int i = begin; i < end; ++i) {
        const auto *b_ = b;
        for (int j = 0; j < n; ++j) {
          const auto dot = internal::bf16_dot_with_fp32_arith(a_, b_, k);
          b_ += ldb;
          c[j*ldc+i] = dot;
        }
        a_ += lda;
      }
    });
    return;
  }
  executorch::extension::parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    const auto *a_ = a + begin * lda;
    for (int i = begin; i < end; ++i) {
      const auto *b_ = b;
      for (int j = 0; j < n; ++j) {
        const auto dot = internal::bf16_dot_with_fp32_arith(a_, b_, k);
        b_ += ldb;
        if (beta == 0) {
          c[j*ldc+i] = alpha*dot;
        } else {
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
        }
      }
      a_ += lda;
    }
  });
}

// Used by custom SDPA's q@k.T; both k-dimensions are already contiguous.
// Serial for the same reason as the gemm_notrans_ specialization above.
template <>
inline void gemm_transa_<torch::executor::BFloat16, float, float>(
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const torch::executor::BFloat16 *a, int64_t lda,
    const torch::executor::BFloat16 *b, int64_t ldb,
    float beta,
    float *c, int64_t ldc) {
  if (n == 1) {
    internal::bf16_gemv_transa_with_fp32_arith(
        m, k, alpha, a, lda, b, beta, c);
    return;
  }

  // Four columns at a time: k is headSize here, short enough that each dot's
  // cross-lane reduction is a large share of its cost, and this tile produces
  // m*n of them.
  const auto *a_ = a;
  for (int64_t i = 0; i < m; ++i) {
    int64_t j = 0;
    for (; j + 4 <= n; j += 4) {
      std::array<float, 4> dots{};
      internal::bf16_dot4_with_fp32_arith(
          a_, b + j * ldb, ldb, k, dots.data());
      for (int64_t d = 0; d < 4; ++d) {
        float *dst = c + (j + d) * ldc + i;
        *dst = (beta == 0) ? alpha * dots[d] : beta * *dst + alpha * dots[d];
      }
    }
    for (; j < n; ++j) {
      const float dot = internal::bf16_dot_with_fp32_arith(a_, b + j * ldb, k);
      float *dst = c + j * ldc + i;
      *dst = (beta == 0) ? alpha * dot : beta * *dst + alpha * dot;
    }
    a_ += lda;
  }
}
// clang-format on

template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b.T)
  for (size_t l = 0; l < k; ++l) {
    for (size_t j = 0; j < n; ++j) {
      opmath_t val = b[j + l * ldb] * alpha;
      int64_t i_m = m / 4;
      for (int64_t i_i = 0; i_i < i_m; ++i_i) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++) {
        c[j * ldc + i] += a[i + l * lda] * val;
      }
    }
  }
}

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t, typename out_t = scalar_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    out_t* c,
    int64_t ldc) {
  // c += alpha * (a @ b.T)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(b[l * ldb + j]);
      });
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = static_cast<out_t>(alpha * dot);
      } else {
        c[j * ldc + i] = static_cast<out_t>(
            beta * static_cast<opmath_t>(c[j * ldc + i]) + alpha * dot);
      }
    }
  }
}

// clang-format off
template <typename scalar_t, typename opmath_t, typename out_t = scalar_t>
void gemm_transab_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    out_t *c, int64_t ldc) {
  // c = beta * c + alpha * (a.T @ b.T)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[i * lda + l]) *
            static_cast<opmath_t>(b[l * ldb + j]);
      });

      if (beta == opmath_t(0)) {
        c[j * ldc + i] = static_cast<out_t>(alpha * dot);
      } else {
        c[j * ldc + i] =
            static_cast<out_t>(beta * static_cast<opmath_t>(c[j * ldc + i]) +
                               alpha * dot);
      }
    }
  }
}
// clang-format on

} // namespace cpublas
} // namespace executorch
