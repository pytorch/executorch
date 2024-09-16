/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/blas/BlasKernel.h>

#ifdef __aarch64__
#include <arm_neon.h>
#include <cpuinfo.h>
#endif

using torch::executor::BFloat16;

namespace executorch {
namespace cpublas {
namespace internal {
#ifdef __aarch64__
static inline float32x4_t f32_fma(float32x4_t a, float32x4_t b, float32x4_t c) {
#ifdef __ARM_FEATURE_FMA
  return vfmaq_f32(a, b, c);
#else
  return vaddq_f32(a, vmulq_f32(b, c));
#endif // __ARM_FEATURE_FMA
}

// The below reduce overload and fp16_dot_with_fp32_arith are adapted
// from llama.cpp's ggml_vec_dot_f32 and surrounding utility
// functions. See NOTE [ GGML Copyright Notice ] above for the
// required notice.

// We need the shift for reduce(), hence the extra constants.
static constexpr auto kF32ElementsPerIterationShift = 5;
static constexpr auto kF32ElementsPerIteration = 1
    << kF32ElementsPerIterationShift;
static_assert(kF32ElementsPerIteration == 32);

static constexpr auto kF32ElementsPerRegisterShift = 2;
static constexpr auto kF32ElementsPerRegister = 1
    << kF32ElementsPerRegisterShift;
static_assert(kF32ElementsPerRegister == 4);

static constexpr auto kF32RegisterPairsPerIteration = 4;
static constexpr auto kF32RegistersPerIteration =
    kF32RegisterPairsPerIteration * 2;
static constexpr auto kF32RegistersPerIterationShift = 3;
static_assert(
    kF32RegistersPerIteration ==
    kF32ElementsPerIteration / kF32ElementsPerRegister);
static_assert(kF32RegistersPerIteration == 1 << kF32RegistersPerIterationShift);

static inline double reduce(float32x4_t x[kF32RegistersPerIteration]) {
  int offset = kF32RegistersPerIteration;
  utils::ForcedUnroll<kF32RegistersPerIterationShift>{}(
      [&offset, &x](auto idx) ET_INLINE_ATTRIBUTE {
        offset /= 2;
        for (int i = 0; i < offset; ++i) {
          x[i] = vaddq_f32(x[i], x[offset + i]);
        }
      });
  return vaddvq_f32(x[0]);
}

static ET_INLINE float32x4_t to_bfloat16(uint16x4_t u16) {
  int32x4_t shift = vdupq_n_s32(16);
  return vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(u16), shift));
}

static ET_INLINE float32x4_t
f32_fma_bf16(float32x4_t a, uint16x4_t b, uint16x4_t c) {
  return f32_fma(a, to_bfloat16(b), to_bfloat16(c));
}

#ifdef __ARM_FEATURE_BF16
static ET_INLINE float32x4_t
f32_dot_bf16(float32x4_t a, bfloat16x8_t b, bfloat16x8_t c) {
  return vbfdotq_f32(a, b, c);
}
#endif // __ARM_FEATURE_BF16

template <bool useBfloat16Dot>
static ET_INLINE void dot_with_fp32_arith_main_inner_loop(
    const BFloat16* vec1,
    const BFloat16* vec2,
    float32x4_t sum[kF32RegistersPerIteration],
    int registerPairIndex) {
#ifdef __ARM_FEATURE_BF16
  if (useBfloat16Dot) {
    const bfloat16x8_t temp_vec1 = vld1q_bf16(reinterpret_cast<const __bf16*>(
        &vec1[registerPairIndex * 2 * kF32ElementsPerRegister]));
    const bfloat16x8_t temp_vec2 = vld1q_bf16(reinterpret_cast<const __bf16*>(
        &vec2[registerPairIndex * 2 * kF32ElementsPerRegister]));
    sum[registerPairIndex] =
        f32_dot_bf16(sum[registerPairIndex], temp_vec1, temp_vec2);
  } else
#endif // __ARM_FEATURE_BF16
  {
    const uint16x8_t temp_vec1 = vld1q_u16(reinterpret_cast<const uint16_t*>(
        &vec1[registerPairIndex * 2 * kF32ElementsPerRegister]));
    const uint16x8_t temp_vec2 = vld1q_u16(reinterpret_cast<const uint16_t*>(
        &vec2[registerPairIndex * 2 * kF32ElementsPerRegister]));

    sum[2 * registerPairIndex] = f32_fma_bf16(
        sum[2 * registerPairIndex],
        vget_low_u16(temp_vec1),
        vget_low_u16(temp_vec2));
    sum[2 * registerPairIndex + 1] = f32_fma_bf16(
        sum[2 * registerPairIndex + 1],
        vget_high_u16(temp_vec1),
        vget_high_u16(temp_vec2));
  }
}

static ET_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop(
    const BFloat16* vec1,
    const BFloat16* vec2,
    float32x4_t* tailSum,
    int idx) {
  const auto temp_vec1 =
      vld1_u16(reinterpret_cast<const uint16_t*>(&vec1[idx]));
  const auto temp_vec2 =
      vld1_u16(reinterpret_cast<const uint16_t*>(&vec2[idx]));
  *tailSum = f32_fma_bf16(*tailSum, temp_vec1, temp_vec2);
}

template <typename T, bool useBfloat16Dot>
float dot_with_fp32_arith(const T* vec1, const T* vec2, int64_t len) {
  float32x4_t sum[kF32RegistersPerIteration] = {vdupq_n_f32(0)};
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned; j += kF32ElementsPerIteration) {
    const auto* vec1_ = vec1 + j;
    const auto* vec2_ = vec2 + j;
    utils::ForcedUnroll<kF32RegisterPairsPerIteration>{}(
        [vec1_, vec2_, &sum](auto k) ET_INLINE_ATTRIBUTE {
          dot_with_fp32_arith_main_inner_loop<useBfloat16Dot>(
              vec1_, vec2_, sum, k);
        });
  }
  auto reducedSum = reduce(sum);

  // First-tier tail fixup: make sure we handle workloads that can
  // benefit from vectorization, but don't fit into our fully unrolled
  // loop above.
  float32x4_t tailSum = vdupq_n_f32(0);
  const auto len_aligned_4 = len & ~3;
  for (int j = len_aligned; j < len_aligned_4; j += 4) {
    dot_with_fp32_arith_vectorized_tail_inner_loop(vec1, vec2, &tailSum, j);
  }
  auto reducedTail = vpaddq_f32(tailSum, tailSum);
  reducedSum += vgetq_lane_f32(vpaddq_f32(reducedTail, reducedTail), 0);

  // Second-tier tail fixup: handle all workloads.
  for (int j = len_aligned_4; j < len; ++j) {
    reducedSum += vec1[j] * vec2[j];
  }
  return reducedSum;
}

float bf16_dot_with_fp32_arith(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t len) {
#ifdef __ARM_FEATURE_BF16
  if (cpuinfo_has_arm_bf16()) {
    return dot_with_fp32_arith<BFloat16, true>(vec1, vec2, len);
  } else
#endif // __ARM_FEATURE_BF16
  {
    return dot_with_fp32_arith<BFloat16, false>(vec1, vec2, len);
  }
}
#endif // __aarch64__
} // namespace internal
} // namespace cpublas
} // namespace executorch
