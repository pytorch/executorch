/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/blas/CPUBlas.h>

#include <limits.h>

#ifdef ET_BUILD_WITH_BLAS
#ifdef ET_BUILD_FOR_APPLE
#include <Accelerate/Accelerate.h>
#else
// clang-format off
extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, const double *a, int *lda, const double *b, int *ldb, double *beta, double *c, int *ldc);
extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc);
// clang-format on
#endif // ET_BUILD_FOR_APPLE
#endif // ET_BUILD_WITH_BLAS

namespace executorch {
namespace cpublas {

using exec_aten::BFloat16;
using exec_aten::Half;

#ifdef ET_BUILD_WITH_BLAS
#ifdef ET_BUILD_FOR_APPLE
inline CBLAS_TRANSPOSE to_cblas_transpose(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose:
      return CblasTrans;
    case TransposeType::NoTranspose:
      return CblasNoTrans;
    case TransposeType::ConjTranspose:
      return CblasConjTrans;
  }
  // Assume no transpose by default
  return CblasNoTrans;
}
#endif // ET_BUILD_FOR_APPLE
#endif // ET_BUILD_WITH_BLAS

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
#ifdef ET_BUILD_FOR_APPLE
  cblas_dgemm(CblasColMajor, to_cblas_transpose(transa), to_cblas_transpose(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
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
#endif // ET_BUILD_FOR_APPLE
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
#ifdef ET_BUILD_FOR_APPLE
  cblas_sgemm(CblasColMajor, to_cblas_transpose(transa), to_cblas_transpose(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
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
#endif // ET_BUILD_FOR_APPLE

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

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Half alpha,
    const Half *a, int64_t lda,
    const Half *b, int64_t ldb,
    const Half beta,
    Half *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);

  using acc_type = utils::compute_dtype<Half>;
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

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const BFloat16 alpha,
    const BFloat16 *a, int64_t lda,
    const BFloat16 *b, int64_t ldb,
    const BFloat16 beta,
    BFloat16 *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);

  using acc_type = utils::compute_dtype<BFloat16>;
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
