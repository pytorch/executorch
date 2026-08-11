/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define TEST_FORALL_SUPPORTED_CTYPES(_, N)   \
  _<double, N>();                            \
  _<float, N>();                             \
  _<int64_t, N>();                           \
  _<uint8_t, N>();                           \
  _<int32_t, N>();                           \
  _<executorch::aten::Half, N>();            \
  _<executorch::aten::BFloat16, N>();        \
  _<executorch::aten::complex<double>, N>(); \
  _<executorch::aten::complex<float>, N>();  \
  _<executorch::aten::complex<executorch::aten::Half>, N>();

namespace {

// Fill a vector with a monotonic sequence of integer values
template <typename T>
void fill_ones(std::vector<T>& arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    arr[i] = static_cast<T>(1);
  }
}

template <typename T>
bool check_all_equal_to(std::vector<T>& arr, const float value) {
  for (size_t i = 0; i < arr.size(); ++i) {
    if (arr[i] != static_cast<T>(value)) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::vector<T> make_values(size_t n, uint32_t seed) {
  std::vector<T> v(n);
  uint32_t state = seed;
  for (size_t i = 0; i < n; ++i) {
    state = state * 1664525u + 1013904223u;
    v[i] = static_cast<T>(
        static_cast<float>(state >> 8) / static_cast<float>(1 << 23) - 1.0f);
  }
  return v;
}

template <typename T>
float reference_dot(const T* a, const T* b, int64_t len) {
  float sum = 0;
  for (int64_t i = 0; i < len; ++i) {
    sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  }
  return sum;
}

// Column-major c = beta * c + alpha * (op(a) @ b), accumulated in fp32, mirror-
// ing the generic gemm_{transa,notrans}_ templates the bf16 specializations
// replace.
template <typename T>
void reference_gemm(
    bool transa,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    const T* a,
    int64_t lda,
    const T* b,
    int64_t ldb,
    float beta,
    std::vector<float>& c,
    int64_t ldc) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float dot = 0;
      for (int64_t l = 0; l < k; ++l) {
        const float av = transa ? static_cast<float>(a[i * lda + l])
                                : static_cast<float>(a[l * lda + i]);
        dot += av * static_cast<float>(b[j * ldb + l]);
      }
      c[j * ldc + i] =
          beta == 0 ? alpha * dot : beta * c[j * ldc + i] + alpha * dot;
    }
  }
}

void expect_near_relative(float actual, float expected, const char* context) {
  EXPECT_NEAR(actual, expected, 1e-4f * std::max(1.0f, std::abs(expected)))
      << context;
}

// Straddle the vectorized main-loop, cleanup-loop and scalar-tail boundaries of
// the bfdot paths: 128 and 32 bf16 per iteration on x86, 32 and 8 on ARM.
constexpr int64_t kDotLengths[] =
    {0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 96, 127, 128, 129, 160, 255, 257};

} // namespace

template <class CTYPE, int64_t N>
void test_matmul_ones() {
  using executorch::cpublas::TransposeType;

  std::vector<CTYPE> in_1(N * N);
  fill_ones(in_1);
  std::vector<CTYPE> in_2(N * N);
  fill_ones(in_2);

  std::vector<CTYPE> out(N * N);

  const CTYPE* in_1_data = in_1.data();
  const CTYPE* in_2_data = in_2.data();

  CTYPE* out_data = out.data();

  // clang-format off
  executorch::cpublas::gemm(
      TransposeType::NoTranspose, TransposeType::NoTranspose,
      N, N, N,
      static_cast<CTYPE>(1),
      in_1_data, N,
      in_2_data, N,
      static_cast<CTYPE>(0),
      out_data, N);
  // clang-format on

  EXPECT_TRUE(check_all_equal_to(out, static_cast<float>(N)));
}

TEST(BlasTest, MatmulOnes) {
  TEST_FORALL_SUPPORTED_CTYPES(test_matmul_ones, 25);
}

// bf16_dot_with_fp32_arith has three implementations -- ARM bfdot, x86
// AVX512-BF16 and the portable fp32 fallback -- selected by compile-time
// support and runtime cpuinfo. Only the one this host dispatches to is covered
// by a given run.
TEST(BlasTest, BF16DotMatchesScalarAccumulation) {
  using torch::executor::BFloat16;

  for (const int64_t len : kDotLengths) {
    const auto a = make_values<BFloat16>(len, 1);
    const auto b = make_values<BFloat16>(len, 2);

    const float actual =
        executorch::cpublas::internal::bf16_dot_with_fp32_arith(
            a.data(), b.data(), len);

    expect_near_relative(
        actual,
        reference_dot(a.data(), b.data(), len),
        ("len=" + std::to_string(len)).c_str());
  }
}

// The bf16-in/float-out gemm specializations used by custom SDPA.
TEST(BlasTest, BF16FloatGemmMatchesScalarAccumulation) {
  using executorch::aten::BFloat16;
  using executorch::cpublas::TransposeType;

  constexpr int64_t kM = 3;
  constexpr int64_t kN = 5;

  for (const bool transa : {false, true}) {
    for (const int64_t k : kDotLengths) {
      if (k == 0) {
        continue;
      }
      // Pad the leading dimensions so a stride bug can't hide behind tightly
      // packed operands.
      const int64_t lda = (transa ? k : kM) + 2;
      const int64_t ldb = k + 3;
      const int64_t ldc = kM + 1;

      const auto a = make_values<BFloat16>(lda * (transa ? kM : k), 3);
      const auto b = make_values<BFloat16>(ldb * kN, 4);

      for (const float alpha : {1.0f, -0.5f}) {
        for (const float beta : {0.0f, 1.0f, 0.25f}) {
          auto c = make_values<float>(ldc * kN, 5);
          auto expected = c;

          // clang-format off
          reference_gemm(
              transa,
              kM, kN, k,
              alpha,
              a.data(), lda,
              b.data(), ldb,
              beta,
              expected, ldc);

          executorch::cpublas::gemm(
              transa ? TransposeType::Transpose : TransposeType::NoTranspose,
              TransposeType::NoTranspose,
              kM, kN, k,
              alpha,
              a.data(), lda,
              b.data(), ldb,
              beta,
              c.data(), ldc);
          // clang-format on

          const std::string context = "transa=" + std::to_string(transa) +
              " k=" + std::to_string(k) + " alpha=" + std::to_string(alpha) +
              " beta=" + std::to_string(beta);
          for (int64_t j = 0; j < kN; ++j) {
            for (int64_t i = 0; i < kM; ++i) {
              expect_near_relative(
                  c[j * ldc + i], expected[j * ldc + i], context.c_str());
            }
          }
        }
      }
    }
  }
}

TEST(BlasTest, BF16FloatGemmDecodeShapesMatchScalarAccumulation) {
  using executorch::aten::BFloat16;
  using executorch::cpublas::TransposeType;

  struct Shape {
    bool transa;
    int64_t m;
    int64_t k;
  };
  constexpr Shape kShapes[] = {
      {false, 64, 511},
      {false, 128, 512},
      {false, 130, 513},
      {true, 512, 64},
      {true, 515, 128},
      {true, 513, 130},
  };

  for (const Shape shape : kShapes) {
    constexpr int64_t kN = 1;
    const int64_t lda = (shape.transa ? shape.k : shape.m) + 3;
    const int64_t ldb = shape.k;
    const int64_t ldc = shape.m;
    const auto a =
        make_values<BFloat16>(lda * (shape.transa ? shape.m : shape.k), 8);
    const auto b = make_values<BFloat16>(shape.k, 9);

    for (const auto [alpha, beta] :
         {std::pair{1.0f, 0.0f}, std::pair{-0.5f, 0.25f}}) {
      auto c = make_values<float>(shape.m, 10);
      auto expected = c;

      // clang-format off
      reference_gemm(
          shape.transa,
          shape.m, kN, shape.k,
          alpha,
          a.data(), lda,
          b.data(), ldb,
          beta,
          expected, ldc);

      executorch::cpublas::gemm(
          shape.transa ? TransposeType::Transpose : TransposeType::NoTranspose,
          TransposeType::NoTranspose,
          shape.m, kN, shape.k,
          alpha,
          a.data(), lda,
          b.data(), ldb,
          beta,
          c.data(), ldc);
      // clang-format on

      const std::string context = "transa=" + std::to_string(shape.transa) +
          " m=" + std::to_string(shape.m) + " k=" + std::to_string(shape.k) +
          " alpha=" + std::to_string(alpha) + " beta=" + std::to_string(beta);
      for (int64_t i = 0; i < shape.m; ++i) {
        expect_near_relative(c[i], expected[i], context.c_str());
      }
    }
  }
}

// beta == 0 must overwrite c rather than read it, so uninitialized garbage in
// the output cannot poison the result.
TEST(BlasTest, BF16FloatGemmBetaZeroIgnoresOutput) {
  using executorch::aten::BFloat16;
  using executorch::cpublas::TransposeType;

  constexpr int64_t kM = 3;
  constexpr int64_t kN = 5;
  constexpr int64_t kK = 40;

  const auto a = make_values<BFloat16>(kK * kM, 6);
  const auto b = make_values<BFloat16>(kK * kN, 7);

  for (const bool transa : {false, true}) {
    std::vector<float> c(kM * kN, std::numeric_limits<float>::quiet_NaN());

    // clang-format off
    executorch::cpublas::gemm(
        transa ? TransposeType::Transpose : TransposeType::NoTranspose,
        TransposeType::NoTranspose,
        kM, kN, kK,
        1.0f,
        a.data(), transa ? kK : kM,
        b.data(), kK,
        0.0f,
        c.data(), kM);
    // clang-format on

    for (const float v : c) {
      EXPECT_FALSE(std::isnan(v)) << "transa=" << transa;
    }
  }
}
