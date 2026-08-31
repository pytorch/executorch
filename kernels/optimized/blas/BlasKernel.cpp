/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This file is mostly the same as
// ReducedPrecisionFloatGemvFastPathKernel.cpp in PyTorch. Actually
// sharing the two versions is a TODO.
#include <executorch/kernels/optimized/blas/BlasKernel.h>
#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/Unroll.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <array>

#ifdef __aarch64__
#include <arm_neon.h>
#include <cpuinfo.h>
#elif defined(__x86_64__) && !defined(_MSC_VER)
#include <cpuinfo.h>
#include <immintrin.h>
#endif

namespace vec = at::vec;
using torch::executor::BFloat16;
using torch::executor::Half;

namespace executorch::cpublas::internal {
constexpr auto kF32RegisterPairsPerIteration = 4;
constexpr auto kF32RegistersPerIteration = kF32RegisterPairsPerIteration * 2;
constexpr auto kF32ElementsPerRegister = vec::Vectorized<float>::size();
constexpr auto kF32ElementsPerIteration =
    kF32RegistersPerIteration * kF32ElementsPerRegister;

namespace {
template <typename T>
constexpr int IntegerLog2(T n, int p = 0) {
  return (n <= 1) ? p : IntegerLog2(n / 2, p + 1);
}

/*
 * NOTE [ GGML Copyright Notice ]
 * The below reduce overload and fp16_dot_with_fp16_arith function is
 * adapted from llama.cpp's ggml_vec_dot_f16 and surrounding utility
 * functions, so here is the required copyright notice:
 *
 * MIT License
 *
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

float reduce(vec::Vectorized<float> x) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE)
  return vaddvq_f32(x);
#else
  return vec::vec_reduce_all<float>(std::plus<vec::Vectorized<float>>(), x);
#endif
}

// The below reduce overload and fp16_dot_with_fp32_arith are adapted
// from llama.cpp's ggml_vec_dot_f32 and surrounding utility
// functions. See NOTE [ GGML Copyright Notice ] above for the
// required notice.
float reduce(vec::VectorizedN<float, kF32RegistersPerIteration>& x) {
  int offset = kF32RegistersPerIteration;
  c10::ForcedUnroll<IntegerLog2(kF32RegistersPerIteration)>{}(
      [&offset, &x](auto idx) {
        offset /= 2;
        for (const auto i : c10::irange(offset)) {
          x[i] = x[i] + x[offset + i];
        }
      });
  return reduce(x[0]);
}

// EXECUTORCH NOTE: removed __ARM_FEATURE_BF16_VECTOR_ARITHMETIC gate
// added in https://github.com/pytorch/pytorch/pull/152766, which I
// complained on.

// We would have to write a separate SVE-specific path to use SVE
// BFDOT. Deferring that for now to get the NEON/ASIMD BFDOT path
// working.
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) && \
    defined(__clang__) && __clang_major__ > 15
// https://godbolt.org/z/z8P4Yncra
#define COMPILER_SUPPORTS_ARM_BF16_TARGET 1
#elif defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) && \
    !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 10
// https://gcc.gnu.org/gcc-10/changes.html
// https://godbolt.org/z/cdGG7vn8o
#define COMPILER_SUPPORTS_ARM_BF16_TARGET 1
#else // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) &&
      // defined(__clang__) && __clang_major__ > 15
#define COMPILER_SUPPORTS_ARM_BF16_TARGET 0
#endif // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) &&
       // defined(__clang__) && __clang_major__ > 15

// GCC 10 / Clang 9 are the first versions with both the target("avx512bf16")
// function attribute and _mm512_dpbf16_ps. clang-cl does not expose the
// AVX-512 intrinsic types unless they are enabled for the whole translation
// unit, so use the portable path for the MSVC ABI.
#if defined(__x86_64__) && !defined(_MSC_VER) && defined(__clang__) && \
    __clang_major__ >= 9
#define COMPILER_SUPPORTS_X86_BF16_TARGET 1
#elif defined(__x86_64__) && !defined(_MSC_VER) && !defined(__clang__) && \
    defined(__GNUC__) && __GNUC__ >= 10
#define COMPILER_SUPPORTS_X86_BF16_TARGET 1
#else
#define COMPILER_SUPPORTS_X86_BF16_TARGET 0
#endif

#if COMPILER_SUPPORTS_ARM_BF16_TARGET
#define TARGET_ARM_BF16_ATTRIBUTE __attribute__((target("arch=armv8.2-a+bf16")))

TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void
dot_with_fp32_arith_main_inner_loop_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
    int registerPairIndex) {
  // NOTE[Intrinsics in bfdot variant]: We can't use
  // vec::Vectorized<BFloat16>::loadu here because linux-aarch64 GCC
  // inexplicably can't convert Vectorized<BFloat16> to
  // bfloat16x8_t. I suspect a bug or incomplete
  // __attribute__((target)) implementation. Intrinsics should be fine
  // because we're using vbfdotq_f32 below anyway.
  const auto temp_vec1 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(
      &vec1[registerPairIndex * vec::Vectorized<BFloat16>::size()]));
  const auto temp_vec2 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(
      &vec2[registerPairIndex * vec::Vectorized<BFloat16>::size()]));
  sum[registerPairIndex] =
      vbfdotq_f32(sum[registerPairIndex], temp_vec1, temp_vec2);
}

TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void
dot_with_fp32_arith_vectorized_tail_inner_loop_bfdot(
    const at::BFloat16* vec1,
    const at::BFloat16* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  // See NOTE[Intrinsics in bfdot variant] above.
  const auto temp_vec1 =
      vld1q_bf16(reinterpret_cast<const bfloat16_t*>(&vec1[idx]));
  const auto temp_vec2 =
      vld1q_bf16(reinterpret_cast<const bfloat16_t*>(&vec2[idx]));
  *tail_sum = vbfdotq_f32(*tail_sum, temp_vec1, temp_vec2);
}

#else
#define TARGET_ARM_BF16_ATTRIBUTE
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET

namespace {

[[maybe_unused]] std::pair<vec::Vectorized<float>, vec::Vectorized<float>>
fmadd(
    const vec::Vectorized<c10::BFloat16>& a,
    const vec::Vectorized<c10::BFloat16>& b,
    const vec::Vectorized<float>& acc_low,
    const vec::Vectorized<float>& acc_high) {
  const auto [a_float_low, a_float_high] = convert_bfloat16_float(a);
  const auto [b_float_low, b_float_high] = convert_bfloat16_float(b);
  return std::make_pair(
      fmadd(a_float_low, b_float_low, acc_low),
      fmadd(a_float_high, b_float_high, acc_high));
}

[[maybe_unused]] vec::Vectorized<float> fmadd(
    const vec::Vectorized<float>& acc,
    const vec::Vectorized<c10::BFloat16>& a,
    const vec::Vectorized<c10::BFloat16>& b) {
  const auto [a_float_low, a_float_high] = convert_bfloat16_float(a);
  const auto [b_float_low, b_float_high] = convert_bfloat16_float(b);
  return fmadd(
      a_float_high, b_float_high, fmadd(a_float_low, b_float_low, acc));
}
} // namespace

template <typename T>
C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_no_bfdot(
    const T* vec1,
    const T* vec2,
    vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
    int registerPairIndex) {
  static_assert(std::is_same_v<T, BFloat16>);
  const auto temp_vec1 = vec::Vectorized<T>::loadu(
      &vec1[registerPairIndex * vec::Vectorized<T>::size()]);
  const auto temp_vec2 = vec::Vectorized<T>::loadu(
      &vec2[registerPairIndex * vec::Vectorized<T>::size()]);

  const auto [result_low, result_high] = fmadd(
      temp_vec1,
      temp_vec2,
      sum[2 * registerPairIndex],
      sum[2 * registerPairIndex + 1]);
  sum[2 * registerPairIndex] = result_low;
  sum[2 * registerPairIndex + 1] = result_high;
}

template <typename T>
C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot(
    const T* vec1,
    const T* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  const auto temp_vec1 = vec::Vectorized<T>::loadu(&vec1[idx]);
  const auto temp_vec2 = vec::Vectorized<T>::loadu(&vec2[idx]);
  *tail_sum = fmadd(*tail_sum, temp_vec1, temp_vec2);
}

template <typename T>
C10_ALWAYS_INLINE auto dot_with_fp32_arith_main_loop_no_bfdot(
    const T* vec1,
    const T* vec2,
    int64_t len) {
  vec::VectorizedN<float, kF32RegistersPerIteration> sum(0);
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned; j += kF32ElementsPerIteration) {
    const auto* vec1_ = vec1 + j;
    const auto* vec2_ = vec2 + j;
    c10::ForcedUnroll<kF32RegisterPairsPerIteration>{}(
        [vec1_, vec2_, &sum](auto k) C10_ALWAYS_INLINE_ATTRIBUTE {
          dot_with_fp32_arith_main_inner_loop_no_bfdot(vec1_, vec2_, sum, k);
        });
  }
  return reduce(sum);
}

#if COMPILER_SUPPORTS_ARM_BF16_TARGET
template <int n>
struct ForcedUnrollTargetBFloat16 {
  template <typename Func>
  TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void operator()(
      const Func& f) const {
    ForcedUnrollTargetBFloat16<n - 1>{}(f);
    f(n - 1);
  }
};

template <>
struct ForcedUnrollTargetBFloat16<1> {
  template <typename Func>
  TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void operator()(
      const Func& f) const {
    f(0);
  }
};

C10_ALWAYS_INLINE TARGET_ARM_BF16_ATTRIBUTE auto
dot_with_fp32_arith_main_loop_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t len) {
  vec::VectorizedN<float, kF32RegistersPerIteration> sum(0);
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned; j += kF32ElementsPerIteration) {
    const auto* vec1_ = vec1 + j;
    const auto* vec2_ = vec2 + j;
    ForcedUnrollTargetBFloat16<kF32RegisterPairsPerIteration>{}(
        [vec1_, vec2_, &sum](auto k)
            C10_ALWAYS_INLINE_ATTRIBUTE TARGET_ARM_BF16_ATTRIBUTE {
              dot_with_fp32_arith_main_inner_loop_bfdot(vec1_, vec2_, sum, k);
            });
  }
  return reduce(sum);
}
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET

static_assert(
    (vec::Vectorized<BFloat16>::size() &
     (vec::Vectorized<BFloat16>::size() - 1)) == 0,
    "Below code expects power-of-2 vector register size!");

// NOTE [GCC code duplication]: The first attempt at landing BFDOT support with
// TARGET_ARM_BF16_ATTRIBUTE failed because unlike clang, GCC will not
// allow inlining a non-bf16-specific function into a bf16-specific
// function. We can work around this by duplicating the code into the
// bfdot and non-bfdot callsites. The code is in this macro to avoid
// actual copy/paste.
#define DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(bfdot_suffix)            \
  /* First-tier tail fixup: make sure we handle workloads that can */          \
  /* benefit from vectorization, but don't fit into our fully unrolled */      \
  /* loop above. */                                                            \
  vec::Vectorized<float> tail_sum(0);                                          \
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);              \
  const auto len_aligned_vec = len & ~(vec::Vectorized<BFloat16>::size() - 1); \
  for (int j = len_aligned; j < len_aligned_vec;                               \
       j += vec::Vectorized<BFloat16>::size()) {                               \
    dot_with_fp32_arith_vectorized_tail_inner_loop##bfdot_suffix(              \
        vec1, vec2, &tail_sum, j);                                             \
  }                                                                            \
  reduced_sum += reduce(tail_sum);                                             \
                                                                               \
  /* Second-tier tail fixup: handle all workloads. */                          \
  for (const auto j : c10::irange(len_aligned_vec, len)) {                     \
    /* Attempting to use Half here caused multiple test failures; */           \
    /* using float to unbreak. (Suspect we need a scalar FMA.) */              \
    float x1 = vec1[j];                                                        \
    float x2 = vec2[j];                                                        \
    reduced_sum += x1 * x2;                                                    \
  }                                                                            \
  return reduced_sum

#if COMPILER_SUPPORTS_ARM_BF16_TARGET
TARGET_ARM_BF16_ATTRIBUTE float dot_with_fp32_arith_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t len) {
  auto reduced_sum = dot_with_fp32_arith_main_loop_bfdot(vec1, vec2, len);
  DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(_bfdot);
}
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET

template <typename T>
C10_ALWAYS_INLINE float
dot_with_fp32_arith_no_bfdot(const T* vec1, const T* vec2, int64_t len) {
  auto reduced_sum = dot_with_fp32_arith_main_loop_no_bfdot(vec1, vec2, len);
  DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(_no_bfdot);
}
#undef DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY

} // namespace

#if COMPILER_SUPPORTS_ARM_BF16_TARGET
// Four dots against a shared vec1: the cross-lane reduction is paid once per
// four results rather than per result, which matters when callers reduce over
// headSize while producing a whole tile of outputs.
TARGET_ARM_BF16_ATTRIBUTE static void dot4_with_fp32_arith_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t stride2,
    int64_t len,
    float* out) {
  constexpr int64_t kElementsPerIteration = 8;
  float32x4_t acc[4] = {
      vdupq_n_f32(0.0f),
      vdupq_n_f32(0.0f),
      vdupq_n_f32(0.0f),
      vdupq_n_f32(0.0f)};
  int64_t idx = 0;
  for (; idx + kElementsPerIteration <= len; idx += kElementsPerIteration) {
    // See NOTE[Intrinsics in bfdot variant] above.
    const auto v1 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(&vec1[idx]));
    for (int64_t j = 0; j < 4; ++j) {
      const auto v2 = vld1q_bf16(
          reinterpret_cast<const bfloat16_t*>(&vec2[j * stride2 + idx]));
      acc[j] = vbfdotq_f32(acc[j], v1, v2);
    }
  }
  const float32x4_t sums =
      vpaddq_f32(vpaddq_f32(acc[0], acc[1]), vpaddq_f32(acc[2], acc[3]));
  vst1q_f32(out, sums);
  for (; idx < len; ++idx) {
    for (int64_t j = 0; j < 4; ++j) {
      out[j] += static_cast<float>(vec1[idx]) *
          static_cast<float>(vec2[j * stride2 + idx]);
    }
  }
}
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET

#if defined(__aarch64__)
template <int64_t kRegisterPairs>
C10_ALWAYS_INLINE void gemv_notrans_block_neon(
    int64_t output_offset,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  float32x4_t acc[kRegisterPairs * 2];
#pragma unroll
  for (int64_t r = 0; r < kRegisterPairs * 2; ++r) {
    if (beta == 0.0f) {
      acc[r] = vdupq_n_f32(0.0f);
    } else {
      acc[r] = vld1q_f32(c + output_offset + r * 4);
      if (beta != 1.0f) {
        acc[r] = vmulq_n_f32(acc[r], beta);
      }
    }
  }

  for (int64_t l = 0; l < k; ++l) {
    const float b_val = static_cast<float>(b[l]) * alpha;
    const auto* a_col =
        reinterpret_cast<const uint16_t*>(a + l * lda + output_offset);
#pragma unroll
    for (int64_t r = 0; r < kRegisterPairs; ++r) {
      const uint16x8_t a_bf16 = vld1q_u16(a_col + r * 8);
      const float32x4_t a_low =
          vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a_bf16), 16));
      const float32x4_t a_high =
          vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(a_bf16), 16));
      acc[r * 2] = vfmaq_n_f32(acc[r * 2], a_low, b_val);
      acc[r * 2 + 1] = vfmaq_n_f32(acc[r * 2 + 1], a_high, b_val);
    }
  }

#pragma unroll
  for (int64_t r = 0; r < kRegisterPairs * 2; ++r) {
    vst1q_f32(c + output_offset + r * 4, acc[r]);
  }
}

static void gemv_notrans_neon(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  int64_t i = 0;
  for (; i + 64 <= m; i += 64) {
    gemv_notrans_block_neon<8>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i + 32 <= m; i += 32) {
    gemv_notrans_block_neon<4>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i + 16 <= m; i += 16) {
    gemv_notrans_block_neon<2>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i + 8 <= m; i += 8) {
    gemv_notrans_block_neon<1>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i < m; ++i) {
    float acc = beta == 0.0f ? 0.0f : beta * c[i];
    for (int64_t l = 0; l < k; ++l) {
      acc += static_cast<float>(a[l * lda + i]) *
          (static_cast<float>(b[l]) * alpha);
    }
    c[i] = acc;
  }
}

static bool use_arm_bf16() {
#if COMPILER_SUPPORTS_ARM_BF16_TARGET
  static const bool supported = cpuinfo_initialize() && cpuinfo_has_arm_bf16();
  return supported;
#else
  return false;
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET
}

static float
platform_bf16_dot(const BFloat16* vec1, const BFloat16* vec2, int64_t len) {
#if COMPILER_SUPPORTS_ARM_BF16_TARGET
  if (use_arm_bf16()) {
    return dot_with_fp32_arith_bfdot(vec1, vec2, len);
  }
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET
  return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
}

#if COMPILER_SUPPORTS_ARM_BF16_TARGET
static bool platform_supports_bf16_dot4() {
  return use_arm_bf16();
}

static void platform_bf16_dot4(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t stride2,
    int64_t len,
    float* out) {
  dot4_with_fp32_arith_bfdot(vec1, vec2, stride2, len, out);
}
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET

static bool platform_supports_bf16_gemv_notrans() {
  return true;
}

static void platform_bf16_gemv_notrans(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  gemv_notrans_neon(m, k, alpha, a, lda, b, beta, c);
}

#if COMPILER_SUPPORTS_ARM_BF16_TARGET
static bool platform_supports_bf16_gemv_transa() {
  return false;
}

static void platform_bf16_gemv_transa(
    int64_t,
    int64_t,
    float,
    const BFloat16*,
    int64_t,
    const BFloat16*,
    float,
    float*) {}
#endif // COMPILER_SUPPORTS_ARM_BF16_TARGET

#elif COMPILER_SUPPORTS_X86_BF16_TARGET

// GCC drops vector type attributes on direct std::array<__m512, ...>
// instantiations and promotes that warning to an error in CI.
struct M512Accumulator {
  __m512 value;
};

// Native x86 bf16 dot using AVX512-BF16's vdpbf16ps (_mm512_dpbf16_ps),
// which computes bf16 x bf16 -> fp32 accumulate in a single instruction.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16"))) static float
dot_with_fp32_arith_x86bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t len) {
  constexpr int kBF16PerRegister = 32;
  constexpr int kAccumulators = 4;
  constexpr int kBF16PerIteration = kBF16PerRegister * kAccumulators;

  std::array<M512Accumulator, kAccumulators> acc{};
  for (int i = 0; i < kAccumulators; ++i) {
    acc[i].value = _mm512_setzero_ps();
  }

  int64_t j = 0;
  const int64_t len_main = len - (len % kBF16PerIteration);
  for (; j < len_main; j += kBF16PerIteration) {
    for (int i = 0; i < kAccumulators; ++i) {
      const int64_t off = j + i * kBF16PerRegister;
      const __m512bh a =
          (__m512bh)_mm512_loadu_si512((const void*)(vec1 + off));
      const __m512bh b =
          (__m512bh)_mm512_loadu_si512((const void*)(vec2 + off));
      acc[i].value = _mm512_dpbf16_ps(acc[i].value, a, b);
    }
  }

  const int64_t len_vec = len - (len % kBF16PerRegister);
  for (; j < len_vec; j += kBF16PerRegister) {
    const __m512bh a = (__m512bh)_mm512_loadu_si512((const void*)(vec1 + j));
    const __m512bh b = (__m512bh)_mm512_loadu_si512((const void*)(vec2 + j));
    acc[0].value = _mm512_dpbf16_ps(acc[0].value, a, b);
  }

  float reduced_sum = 0;
  for (int i = 0; i < kAccumulators; ++i) {
    reduced_sum += _mm512_reduce_add_ps(acc[i].value);
  }
  for (; j < len; ++j) {
    const float x1 = vec1[j];
    const float x2 = vec2[j];
    reduced_sum += x1 * x2;
  }
  return reduced_sum;
}

// x86 counterpart of dot4_with_fp32_arith_bfdot. The single-dot path above
// carries four accumulators for ILP, but at len == headSize it never enters
// its 128-wide main loop: it fills one accumulator and then reduces all four,
// so three of every four reductions are over zeros. Here each accumulator
// holds a distinct output instead, so the same four reductions do four times
// the work.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16"))) static void
dot4_with_fp32_arith_x86bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t stride2,
    int64_t len,
    float* out) {
  constexpr int64_t kBF16PerRegister = 32;
  std::array<M512Accumulator, 4> acc{};
  for (int i = 0; i < 4; ++i) {
    acc[i].value = _mm512_setzero_ps();
  }

  int64_t j = 0;
  const int64_t len_vec = len - (len % kBF16PerRegister);
  for (; j < len_vec; j += kBF16PerRegister) {
    const __m512bh a = (__m512bh)_mm512_loadu_si512((const void*)(vec1 + j));
    for (int i = 0; i < 4; ++i) {
      const __m512bh b =
          (__m512bh)_mm512_loadu_si512((const void*)(vec2 + i * stride2 + j));
      acc[i].value = _mm512_dpbf16_ps(acc[i].value, a, b);
    }
  }

  for (int i = 0; i < 4; ++i) {
    out[i] = _mm512_reduce_add_ps(acc[i].value);
  }

  // Scalar fp32 tail, matching the numerics of the no_bfdot path.
  for (; j < len; ++j) {
    const float x1 = vec1[j];
    for (int i = 0; i < 4; ++i) {
      out[i] += x1 * static_cast<float>(vec2[i * stride2 + j]);
    }
  }
}

template <int64_t kRegisters>
__attribute__((
    target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) C10_ALWAYS_INLINE void
gemv_notrans_block_x86bf16(
    int64_t output_offset,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  std::array<M512Accumulator, kRegisters> acc{};
  for (int64_t r = 0; r < kRegisters; ++r) {
    if (beta == 0.0f) {
      acc[r].value = _mm512_setzero_ps();
    } else {
      acc[r].value = _mm512_loadu_ps(c + output_offset + r * 16);
      if (beta != 1.0f) {
        acc[r].value = _mm512_mul_ps(acc[r].value, _mm512_set1_ps(beta));
      }
    }
  }

  for (int64_t l = 0; l < k; ++l) {
    const __m512 b_vec = _mm512_set1_ps(static_cast<float>(b[l]) * alpha);
    const BFloat16* a_col = a + l * lda + output_offset;
    for (int64_t r = 0; r < kRegisters; ++r) {
      const __m256i a_bf16 =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_col + r * 16));
      // GCC 10 and 11 support AVX512-BF16 dot products but do not expose
      // _mm512_cvtpbh_ps. Widen the bit patterns with AVX-512F instead.
      const __m512 a_vec = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bf16), 16));
      acc[r].value = _mm512_fmadd_ps(a_vec, b_vec, acc[r].value);
    }
  }

  for (int64_t r = 0; r < kRegisters; ++r) {
    _mm512_storeu_ps(c + output_offset + r * 16, acc[r].value);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,fma"))) static void
gemv_notrans_x86bf16(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  int64_t i = 0;
  for (; i + 128 <= m; i += 128) {
    gemv_notrans_block_x86bf16<8>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i + 64 <= m; i += 64) {
    gemv_notrans_block_x86bf16<4>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i + 32 <= m; i += 32) {
    gemv_notrans_block_x86bf16<2>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i + 16 <= m; i += 16) {
    gemv_notrans_block_x86bf16<1>(i, k, alpha, a, lda, b, beta, c);
  }
  for (; i < m; ++i) {
    float acc = beta == 0.0f ? 0.0f : beta * c[i];
    for (int64_t l = 0; l < k; ++l) {
      acc += static_cast<float>(a[l * lda + i]) *
          (static_cast<float>(b[l]) * alpha);
    }
    c[i] = acc;
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16"))) static void
gemv_transa_x86bfdot(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  int64_t i = 0;
  for (; i + 4 <= m; i += 4) {
    std::array<float, 4> dots{};
    dot4_with_fp32_arith_x86bfdot(b, a + i * lda, lda, k, dots.data());
    for (int64_t d = 0; d < 4; ++d) {
      c[i + d] =
          beta == 0.0f ? alpha * dots[d] : beta * c[i + d] + alpha * dots[d];
    }
  }
  for (; i < m; ++i) {
    const float dot = dot_with_fp32_arith_x86bfdot(a + i * lda, b, k);
    c[i] = beta == 0.0f ? alpha * dot : beta * c[i] + alpha * dot;
  }
}

static bool use_x86_bf16() {
  static const bool supported =
      cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
  return supported;
}

static float
platform_bf16_dot(const BFloat16* vec1, const BFloat16* vec2, int64_t len) {
  return use_x86_bf16() ? dot_with_fp32_arith_x86bfdot(vec1, vec2, len)
                        : dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
}

static bool platform_supports_bf16_dot4() {
  return use_x86_bf16();
}

static void platform_bf16_dot4(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t stride2,
    int64_t len,
    float* out) {
  dot4_with_fp32_arith_x86bfdot(vec1, vec2, stride2, len, out);
}

static bool platform_supports_bf16_gemv_notrans() {
  return use_x86_bf16();
}

static void platform_bf16_gemv_notrans(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  gemv_notrans_x86bf16(m, k, alpha, a, lda, b, beta, c);
}

static bool platform_supports_bf16_gemv_transa() {
  return use_x86_bf16();
}

static void platform_bf16_gemv_transa(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
  gemv_transa_x86bfdot(m, k, alpha, a, lda, b, beta, c);
}

#else

static float
platform_bf16_dot(const BFloat16* vec1, const BFloat16* vec2, int64_t len) {
  return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
}

#endif // defined(__aarch64__)

float bf16_dot_with_fp32_arith(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t len) {
  return platform_bf16_dot(vec1, vec2, len);
}

void bf16_dot4_with_fp32_arith(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t stride2,
    int64_t len,
    float* out) {
#if COMPILER_SUPPORTS_ARM_BF16_TARGET || COMPILER_SUPPORTS_X86_BF16_TARGET
  if (platform_supports_bf16_dot4()) {
    platform_bf16_dot4(vec1, vec2, stride2, len, out);
    return;
  }
#endif
  for (int64_t j = 0; j < 4; ++j) {
    out[j] = bf16_dot_with_fp32_arith(vec1, vec2 + j * stride2, len);
  }
}

void bf16_gemv_notrans_with_fp32_arith(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
#if defined(__aarch64__) || COMPILER_SUPPORTS_X86_BF16_TARGET
  if (platform_supports_bf16_gemv_notrans()) {
    platform_bf16_gemv_notrans(m, k, alpha, a, lda, b, beta, c);
    return;
  }
#endif
  if (beta == 0.0f) {
    std::fill(c, c + m, 0.0f);
  } else if (beta != 1.0f) {
    for (int64_t i = 0; i < m; ++i) {
      c[i] *= beta;
    }
  }
  for (int64_t l = 0; l < k; ++l) {
    const BFloat16* a_col = a + l * lda;
    const float b_val = static_cast<float>(b[l]) * alpha;
    for (int64_t i = 0; i < m; ++i) {
      c[i] += static_cast<float>(a_col[i]) * b_val;
    }
  }
}

void bf16_gemv_transa_with_fp32_arith(
    int64_t m,
    int64_t k,
    float alpha,
    const BFloat16* a,
    int64_t lda,
    const BFloat16* b,
    float beta,
    float* c) {
#if COMPILER_SUPPORTS_ARM_BF16_TARGET || COMPILER_SUPPORTS_X86_BF16_TARGET
  if (platform_supports_bf16_gemv_transa()) {
    platform_bf16_gemv_transa(m, k, alpha, a, lda, b, beta, c);
    return;
  }
#endif
  int64_t i = 0;
  for (; i + 4 <= m; i += 4) {
    std::array<float, 4> dots{};
    bf16_dot4_with_fp32_arith(b, a + i * lda, lda, k, dots.data());
    for (int64_t d = 0; d < 4; ++d) {
      c[i + d] =
          beta == 0.0f ? alpha * dots[d] : beta * c[i + d] + alpha * dots[d];
    }
  }
  for (; i < m; ++i) {
    const float dot = bf16_dot_with_fp32_arith(a + i * lda, b, k);
    c[i] = beta == 0.0f ? alpha * dot : beta * c[i] + alpha * dot;
  }
}

} // namespace executorch::cpublas::internal
