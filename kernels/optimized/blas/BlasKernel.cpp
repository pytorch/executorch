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

#ifdef __aarch64__
#include <arm_neon.h>
#include <cpuinfo.h>
#endif

namespace vec = at::vec;
using executorch::extension::parallel_for;
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
#define COMPILER_SUPPORTS_BF16_TARGET 1
#elif defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) && \
    !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 10
// https://gcc.gnu.org/gcc-10/changes.html
// https://godbolt.org/z/cdGG7vn8o
#define COMPILER_SUPPORTS_BF16_TARGET 1
#else // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) &&
      // defined(__clang__) && __clang_major__ > 15
#define COMPILER_SUPPORTS_BF16_TARGET 0
#endif // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) &&
       // defined(__clang__) && __clang_major__ > 15

#if COMPILER_SUPPORTS_BF16_TARGET
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
#endif // COMPILER_SUPPORTS_BF16_TARGET

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

#if COMPILER_SUPPORTS_BF16_TARGET
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
#endif // COMPILER_SUPPORTS_BF16_TARGET

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

#if COMPILER_SUPPORTS_BF16_TARGET
TARGET_ARM_BF16_ATTRIBUTE float dot_with_fp32_arith_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t len) {
  auto reduced_sum = dot_with_fp32_arith_main_loop_bfdot(vec1, vec2, len);
  DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(_bfdot);
}
#endif // COMPILER_SUPPORTS_BF16_TARGET

template <typename T>
C10_ALWAYS_INLINE float
dot_with_fp32_arith_no_bfdot(const T* vec1, const T* vec2, int64_t len) {
  auto reduced_sum = dot_with_fp32_arith_main_loop_no_bfdot(vec1, vec2, len);
  DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(_no_bfdot);
}
#undef DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY

} // namespace

float bf16_dot_with_fp32_arith(
    const at::BFloat16* vec1,
    const at::BFloat16* vec2,
    int64_t len) {
#if COMPILER_SUPPORTS_BF16_TARGET
  if (cpuinfo_has_arm_bf16()) {
    return dot_with_fp32_arith_bfdot(vec1, vec2, len);
  } else
#endif // COMPILER_SUPPORTS_BF16_TARGET
  {
    return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
  }
}

} // namespace executorch::cpublas::internal
