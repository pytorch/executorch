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
extern "C" void cgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
extern "C" void zgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
// clang-format on
#endif // ET_BUILD_FOR_APPLE
#endif // ET_BUILD_WITH_BLAS

#if defined(ET_BUILD_WITH_BLAS) && !defined(ET_BUILD_FOR_APPLE)
#if defined(__linux__) && defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/threadpool_guard.h>
#endif // defined(__linux__) && defined(ET_USE_THREADPOOL)

// Some host BLAS backends (notably MKL) parallelize gemm internally with their
// own OpenMP thread team. ExecuTorch already parallelizes operators across its
// own threadpool, and kernels such as the optimized SDPA call gemm from inside
// a pthreadpool worker thread. Letting the BLAS spin up a nested OpenMP team
// from that worker crashes on Linux x86 hosts (SEGV in __kmp_create_worker /
// KMP_UBER_GTID).
//
// When a NoThreadPoolGuard is active on this thread, force the BLAS
// single-threaded for the duration of the gemm call and restore afterwards.
// ExecuTorch enables the guard on its threadpool workers (the nested case);
// top-level callers may also enable it to force single-threaded execution.
//
// The whole mechanism is gated on defined(__linux__) &&
// defined(ET_USE_THREADPOOL):
//  - ET_USE_THREADPOOL: without ExecuTorch's threadpool there are no worker
//    threads to nest from, so there is nothing to constrain -- and we avoid any
//    dependency on the threadpool extension (NoThreadPoolGuard) in BLAS builds
//    that do not link it.
//  - __linux__: the nested-OpenMP crash is specific to the Linux x86 host
//    iomp5/MKL BLAS path, and the "undefined weak symbol resolves to null"
//    behavior the steering relies on is ELF-specific. On macOS/Mach-O a plain
//    weak undefined symbol fails to link, and Windows is MSVC (no weak
//    symbols), so on both the guard must compile to a no-op.
//
// The OpenMP nthreads-var ICV written by omp_set_num_threads / read by
// omp_get_max_threads is per-thread (one copy per data environment per the
// OpenMP spec), so each threadpool worker captures and restores its own value;
// no cross-thread synchronization is needed. The symbols are declared weak so
// this steers the OpenMP runtime the BLAS has already loaded (e.g. MKL's iomp5)
// WITHOUT compiling this translation unit with -fopenmp (which could link a
// second, conflicting OpenMP runtime); when no OpenMP-threaded BLAS is linked
// they stay null and the guard is a no-op (e.g. OSS Eigen).
#if defined(__linux__) && defined(ET_USE_THREADPOOL)
extern "C" __attribute__((weak)) int omp_get_max_threads(void);
extern "C" __attribute__((weak)) void omp_set_num_threads(int);
#endif // defined(__linux__) && defined(ET_USE_THREADPOOL)

namespace {
class ScopedSingleThreadBlas {
 public:
  ScopedSingleThreadBlas() {
#if defined(__linux__) && defined(ET_USE_THREADPOOL)
    // Only constrain the BLAS when a NoThreadPoolGuard is active on this
    // thread; otherwise leave gemm free to use the threaded BLAS.
    if (!::executorch::extension::threadpool::NoThreadPoolGuard::is_enabled()) {
      return;
    }
    if (omp_get_max_threads != nullptr && omp_set_num_threads != nullptr) {
      prev_num_threads_ = omp_get_max_threads();
      if (prev_num_threads_ > 1) {
        omp_set_num_threads(1);
        restore_ = true;
      }
    }
#endif // defined(__linux__) && defined(ET_USE_THREADPOOL)
  }
  ~ScopedSingleThreadBlas() {
#if defined(__linux__) && defined(ET_USE_THREADPOOL)
    if (restore_) {
      omp_set_num_threads(prev_num_threads_);
    }
#endif // defined(__linux__) && defined(ET_USE_THREADPOOL)
  }
  ScopedSingleThreadBlas(const ScopedSingleThreadBlas&) = delete;
  ScopedSingleThreadBlas& operator=(const ScopedSingleThreadBlas&) = delete;
  ScopedSingleThreadBlas(ScopedSingleThreadBlas&&) = delete;
  ScopedSingleThreadBlas& operator=(ScopedSingleThreadBlas&&) = delete;

 private:
  [[maybe_unused]] int prev_num_threads_ = 1;
  [[maybe_unused]] bool restore_ = false;
};
} // namespace
#endif // defined(ET_BUILD_WITH_BLAS) && !defined(ET_BUILD_FOR_APPLE)

namespace executorch {
namespace cpublas {

using executorch::aten::BFloat16;
using executorch::aten::complex;
using executorch::aten::Half;

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
  // See note above: avoid a nested OpenMP team when called from inside an
  // ExecuTorch threadpool worker.
  ScopedSingleThreadBlas single_thread_blas;
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
  // Avoid a nested OpenMP team if the BLAS (e.g. MKL) is multithreaded and we
  // are already running inside an ExecuTorch threadpool worker. See note above.
  ScopedSingleThreadBlas single_thread_blas;
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

// clang-format off
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const complex<double> alpha,
    const complex<double> *a, int64_t lda,
    const complex<double> *b, int64_t ldb,
    const complex<double> beta,
    complex<double> *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if defined(ET_BUILD_WITH_BLAS) && !defined(ET_BUILD_FOR_APPLE)
  // See note above: avoid a nested OpenMP team when called from inside an
  // ExecuTorch threadpool worker.
  ScopedSingleThreadBlas single_thread_blas;
  complex<double> alpha_ = alpha, beta_ = beta;
  int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
  char transa_ = to_blas(transa), transb_ = to_blas(transb);
  zgemm_(
      &transa_, &transb_,
      &m_, &n_, &k_,
      &alpha_,
      a, &lda_,
      b, &ldb_,
      &beta_,
      c, &ldc_);
#else
  using acc_type = utils::compute_dtype<complex<double>>;
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
    const complex<float> alpha,
    const complex<float> *a, int64_t lda,
    const complex<float> *b, int64_t ldb,
    const complex<float> beta,
    complex<float> *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
  #if defined(ET_BUILD_WITH_BLAS) && !defined(ET_BUILD_FOR_APPLE)
  // See note above: avoid a nested OpenMP team when called from inside an
  // ExecuTorch threadpool worker.
  ScopedSingleThreadBlas single_thread_blas;
  complex<float> alpha_ = alpha, beta_ = beta;
  int m_ = m, n_ = n, k_ = k, lda_ = lda, ldb_ = ldb, ldc_ = ldc;
  char transa_ = to_blas(transa), transb_ = to_blas(transb);
  cgemm_(
      &transa_, &transb_,
      &m_, &n_, &k_,
      &alpha_,
      a, &lda_,
      b, &ldb_,
      &beta_,
      c, &ldc_);
#else
  using acc_type = utils::compute_dtype<complex<float>>;
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
    const complex<Half> alpha,
    const complex<Half> *a, int64_t lda,
    const complex<Half> *b, int64_t ldb,
    const complex<Half> beta,
    complex<Half> *c, int64_t ldc) {
  normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);

  using acc_type = utils::compute_dtype<complex<Half>>;
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
