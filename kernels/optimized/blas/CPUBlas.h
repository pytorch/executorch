/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include <executorch/kernels/optimized/blas/BlasKernel.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace executorch {
namespace cpublas {

using BFloat16 = torch::executor::BFloat16;
using Half = torch::executor::Half;

enum class TransposeType {
  NoTranspose,
  Transpose,
  ConjTranspose,
};

// clang-format off
void normalize_last_dims(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t *lda, int64_t *ldb, int64_t *ldc);
// clang-format on

inline char to_blas(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose:
      return 'T';
    case TransposeType::NoTranspose:
      return 'N';
    case TransposeType::ConjTranspose:
      return 'C';
  }
  // Assume no transpose by default
  return 'N';
}

// clang-format off
template <typename scalar_t, typename opmath_t>
void gemm_impl(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  if (transa == TransposeType::NoTranspose &&
      transb == TransposeType::NoTranspose) {
    return gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if (
      transa == TransposeType::Transpose &&
      transb != TransposeType::Transpose) {
    gemm_transa_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if (
      transa == TransposeType::NoTranspose &&
      transb == TransposeType::Transpose) {
    gemm_transb_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else { // transa == TransposeType::Transpose && transb ==
           // TransposeType::Transpose
    gemm_transab_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}
// clang-format on

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    double beta,
    double *c, int64_t ldc);
// clang-format on

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc);
// clang-format on

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Half alpha,
    const Half *a, int64_t lda,
    const Half *b, int64_t ldb,
    const Half beta,
    Half *c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const BFloat16 alpha,
    const BFloat16 *a, int64_t lda,
    const BFloat16 *b, int64_t ldb,
    const BFloat16 beta,
    BFloat16 *c, int64_t ldc);
// clang-format on

// clang-format off
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const T alpha,
    const T *a, int64_t lda,
    const T *b, int64_t ldb,
    const T beta,
    T *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);

  using acc_type = utils::compute_dtype<T>;
  gemm_impl(
      transa, transb,
      m, n, k,
      static_cast<const acc_type>(alpha),
      a, lda,
      b, ldb,
      static_cast<const acc_type>(beta),
      c, ldc);
}
// clang-format on

} // namespace cpublas
} // namespace executorch
