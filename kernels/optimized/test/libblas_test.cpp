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

#include <array>
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

void reference_bf16_fp32_gemv_notrans(
    int64_t m,
    int64_t k,
    float alpha,
    const executorch::aten::BFloat16* a,
    int64_t lda,
    const float* b,
    float beta,
    std::vector<float>& c) {
  for (int64_t i = 0; i < m; ++i) {
    float dot = 0.0f;
    for (int64_t l = 0; l < k; ++l) {
      dot += static_cast<float>(a[l * lda + i]) * b[l];
    }
    c[i] = beta == 0.0f ? alpha * dot : beta * c[i] + alpha * dot;
  }
}

// Column-major c = beta * c + alpha * (op(a) @ op(b)), accumulated in fp32.
template <typename T>
void reference_gemm(
    bool transa,
    bool transb,
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
        const float bv = transb ? static_cast<float>(b[l * ldb + j])
                                : static_cast<float>(b[j * ldb + l]);
        dot += av * bv;
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
constexpr std::array<int64_t, 18> kDotLengths{
    0,
    1,
    7,
    8,
    15,
    16,
    31,
    32,
    33,
    63,
    64,
    96,
    127,
    128,
    129,
    160,
    255,
    257};

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
              false,
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
  constexpr std::array<Shape, 7> kShapes{
      Shape{false, 64, 511},
      Shape{false, 96, 511},
      Shape{false, 128, 512},
      Shape{false, 130, 513},
      Shape{true, 512, 64},
      Shape{true, 515, 128},
      Shape{true, 513, 130},
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
          false,
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

TEST(BlasTest, BF16FP32GemvDecodeShapesMatchScalarAccumulation) {
  using executorch::aten::BFloat16;

  for (const auto [m, k] :
       {std::pair{64, 511}, std::pair{128, 512}, std::pair{130, 513}}) {
    const int64_t lda = m + 3;
    const auto a = make_values<BFloat16>(lda * k, 11);
    const auto b = make_values<float>(k, 12);

    for (const auto [alpha, beta] :
         {std::pair{1.0f, 0.0f}, std::pair{-0.5f, 0.25f}}) {
      auto c = make_values<float>(m, 13);
      auto expected = c;

      reference_bf16_fp32_gemv_notrans(
          m, k, alpha, a.data(), lda, b.data(), beta, expected);
      executorch::cpublas::gemv(
          m, k, alpha, a.data(), lda, b.data(), beta, c.data());

      const std::string context = "m=" + std::to_string(m) +
          " k=" + std::to_string(k) + " alpha=" + std::to_string(alpha) +
          " beta=" + std::to_string(beta);
      for (int64_t i = 0; i < m; ++i) {
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

  constexpr int64_t kLdc = kM + 2;
  constexpr float kPadding = 123.0f;
  const auto a = make_values<BFloat16>(kK * kM, 6);
  const auto b = make_values<BFloat16>(kK * kN, 7);

  for (const bool transa : {false, true}) {
    for (const bool transb : {false, true}) {
      for (const float alpha : {1.0f, -0.5f, 0.0f}) {
        SCOPED_TRACE(
            ::testing::Message()
            << "m=" << kM << " n=" << kN << " k=" << kK << " transa=" << transa
            << " transb=" << transb << " alpha=" << alpha << " beta=0");
        std::vector<float> c(kLdc * kN, kPadding);
        for (int64_t j = 0; j < kN; ++j) {
          for (int64_t i = 0; i < kM; ++i) {
            c[j * kLdc + i] = std::numeric_limits<float>::quiet_NaN();
          }
        }
        auto expected = c;
        const int64_t lda = transa ? kK : kM;
        const int64_t ldb = transb ? kN : kK;
        reference_gemm(
            transa,
            transb,
            kM,
            kN,
            kK,
            alpha,
            a.data(),
            lda,
            b.data(),
            ldb,
            0.0f,
            expected,
            kLdc);

        executorch::cpublas::gemm(
            transa ? TransposeType::Transpose : TransposeType::NoTranspose,
            transb ? TransposeType::Transpose : TransposeType::NoTranspose,
            kM,
            kN,
            kK,
            alpha,
            a.data(),
            lda,
            b.data(),
            ldb,
            0.0f,
            c.data(),
            kLdc);

        for (size_t index = 0; index < c.size(); ++index) {
          SCOPED_TRACE(::testing::Message() << "index=" << index);
          expect_near_relative(c[index], expected[index], "BF16 GEMM");
        }
      }
    }
  }
}

void test_bfloat16_inputs_accumulate_into_float(
    const executorch::cpublas::TransposeType transa,
    const executorch::cpublas::TransposeType transb,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const float alpha,
    const float beta,
    const int64_t output_padding) {
  using executorch::aten::BFloat16;
  using executorch::cpublas::TransposeType;

  const int64_t ldc = m + output_padding;
  constexpr float padding = 123.0f;
  SCOPED_TRACE(
      ::testing::Message() << "m=" << m << " n=" << n << " k=" << k
                           << " transa=" << static_cast<int>(transa)
                           << " transb=" << static_cast<int>(transb)
                           << " alpha=" << alpha << " beta=" << beta
                           << " ldc=" << ldc);
  const int64_t lda = transa == TransposeType::NoTranspose ? m + 2 : k + 2;
  const int64_t ldb = transb == TransposeType::NoTranspose ? k + 1 : n + 1;
  const int64_t a_columns = transa == TransposeType::NoTranspose ? k : m;
  const int64_t b_columns = transb == TransposeType::NoTranspose ? n : k;

  const auto a = make_values<BFloat16>(lda * a_columns, 14);
  const auto b = make_values<BFloat16>(ldb * b_columns, 15);

  // Check stores past the last column as well as padding within each column.
  constexpr int64_t kGuardElements = 64;
  std::vector<float> out(ldc * n + kGuardElements, padding);
  std::vector<float> expected = out;
  reference_gemm(
      transa == TransposeType::Transpose,
      transb == TransposeType::Transpose,
      m,
      n,
      k,
      alpha,
      a.data(),
      lda,
      b.data(),
      ldb,
      beta,
      expected,
      ldc);

  executorch::cpublas::gemm(
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a.data(),
      lda,
      b.data(),
      ldb,
      beta,
      out.data(),
      ldc);

  for (size_t index = 0; index < out.size(); ++index) {
    if (index >= static_cast<size_t>(ldc * n) || index % ldc >= m) {
      EXPECT_EQ(out[index], padding) << "index=" << index;
    } else {
      EXPECT_NEAR(out[index], expected[index], 1e-3f) << "index=" << index;
    }
  }
}

TEST(BlasTest, BFloat16InputsAccumulateIntoFloatAcrossLayouts) {
  using executorch::cpublas::TransposeType;

  for (const auto shape :
       {std::array<int64_t, 3>{1, 4, 1},
        std::array<int64_t, 3>{5, 1, 7},
        std::array<int64_t, 3>{5, 2, 7},
        std::array<int64_t, 3>{5, 3, 7},
        std::array<int64_t, 3>{5, 4, 7},
        std::array<int64_t, 3>{5, 5, 7},
        std::array<int64_t, 3>{12, 8, 4},
        std::array<int64_t, 3>{13, 9, 17},
        std::array<int64_t, 3>{25, 17, 33}}) {
    for (const auto transa :
         {TransposeType::NoTranspose, TransposeType::Transpose}) {
      for (const auto transb :
           {TransposeType::NoTranspose, TransposeType::Transpose}) {
        for (const auto scales :
             {std::array<float, 2>{1.0f, 0.0f},
              std::array<float, 2>{-0.5f, 0.0f},
              std::array<float, 2>{0.0f, 0.0f},
              std::array<float, 2>{1.0f, 1.0f},
              std::array<float, 2>{0.75f, -0.25f}}) {
          for (const int64_t output_padding : {0, 2}) {
            test_bfloat16_inputs_accumulate_into_float(
                transa,
                transb,
                shape[0],
                shape[1],
                shape[2],
                scales[0],
                scales[1],
                output_padding);
          }
        }
      }
    }
  }
}

TEST(BlasTest, BFloat16InputsAccumulateIntoFloatLargeThenSmall) {
  using executorch::cpublas::TransposeType;

  // Cross the product scratch cache limit, then reuse the same thread.
  for (const int64_t size : {5, 513, 5}) {
    test_bfloat16_inputs_accumulate_into_float(
        TransposeType::NoTranspose,
        TransposeType::NoTranspose,
        size,
        size,
        7,
        0.75f,
        -0.25f,
        0);
  }
}

TEST(BlasTest, BFloat16SmallColumnCountsUseFallback) {
  using executorch::cpublas::TransposeType;

  for (const auto transb :
       {TransposeType::NoTranspose, TransposeType::Transpose}) {
    for (const int64_t n : {1, 2, 3}) {
      EXPECT_FALSE(executorch::cpublas::gemm_uses_kleidiai_bfloat16(transb, n))
          << "transb=" << static_cast<int>(transb) << " n=" << n;
    }
  }
}
