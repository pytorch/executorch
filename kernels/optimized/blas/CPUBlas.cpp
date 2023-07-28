/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/blas/CPUBlas.h>

#ifdef ET_BUILD_WITH_BLAS
// clang-format off
extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, const double *a, int *lda, const double *b, int *ldb, double *beta, double *c, int *ldc);
extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);
// clang-format on
#endif

namespace executorch {
namespace cpublas {

// clang-format off
void normalize_last_dims(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t *lda, int64_t *ldb, int64_t *ldc) {
  if (n == 1) {
    *ldc = m;
  }

  if(transa != TransposeType::NoTranspose) {
    if (m == 1) {
      *lda = k;
    }
  } else if(k == 1) {
    *lda = m;
  }

  if(transb != TransposeType::NoTranspose) {
    if (k == 1) {
      *ldb = n;
    }
  } else if (n == 1) {
    *ldb = k;
  }
}
// clang-format on

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const double alpha,
    const double *a, int64_t lda,
    const double *b, int64_t ldb,
    const double beta,
    double *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#ifdef ET_BUILD_WITH_BLAS
  int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
  double alpha_ = alpha, beta_ = beta;
  char transa_ = to_blas(transa), transb_ = to_blas(transb);
  dgemm_(
      &transa_, &transb_,
      &m_, &n_, &k_,
      &alpha_,
      a, &lda_,
      b, &ldb_,
      &beta_,
      c, &ldc_);
#else
  using acc_type = utils::compute_dtype<float>;
  gemm_impl(
      transa, transb,
      m, n, k,
      static_cast<const acc_type>(alpha),
      a, lda,
      b, ldb,
      static_cast<const acc_type>(beta),
      c, ldc);
#endif
}
// clang-format on

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const float alpha,
    const float *a, int64_t lda,
    const float *b, int64_t ldb,
    const float beta,
    float *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#ifdef ET_BUILD_WITH_BLAS
  int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
  float alpha_ = alpha, beta_ = beta;
  char transa_ = to_blas(transa), transb_ = to_blas(transb);
  sgemm_(
      &transa_, &transb_,
      &m_, &n_, &k_,
      &alpha_,
      a, &lda_,
      b, &ldb_,
      &beta_,
      c, &ldc_);
#else
  using acc_type = utils::compute_dtype<float>;
  gemm_impl(
      transa, transb,
      m, n, k,
      static_cast<const acc_type>(alpha),
      a, lda,
      b, ldb,
      static_cast<const acc_type>(beta),
      c, ldc);
#endif
}
// clang-format on

} // namespace cpublas
} // namespace executorch
