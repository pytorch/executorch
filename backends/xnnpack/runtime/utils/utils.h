/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <unistd.h>
#include <cmath>
#include <limits>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace executorch {
namespace backends {
namespace xnnpack {
namespace utils {

struct QuantizationParams {
  double scale;
  int32_t zero_point;
};

executorch::runtime::Error ChooseQuantizationParams(
    float min,
    float max,
    int32_t qmin,
    int32_t qmax,
    QuantizationParams& result,
    bool preserve_sparsity,
    bool force_scale_power_of_two,
    bool reduce_range);

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int64_t qvalue;
  constexpr int64_t qmin = std::numeric_limits<T>::min();
  constexpr int64_t qmax = std::numeric_limits<T>::max();
  float inv_scale = 1.0f / static_cast<float>(scale);
  qvalue = static_cast<int64_t>(zero_point + Round(value * inv_scale));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

#ifdef __aarch64__
template <typename Tx8>
Tx8 vqmov(int16x8_t vraw);

template <typename T, typename Tx8>
void vst1(T* out, Tx8 vout);

template <typename underlying_t, typename underlying_x8_t>
void quantize_tensor_arm64_q8(
    const float* __restrict__ in,
    underlying_t* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point) {
  const float inv_scale = 1.0f / scale;
  uint32_t i = 0;
  underlying_t* out_underlying = reinterpret_cast<underlying_t*>(out);
  const float32x4_t vinv_scale = vdupq_n_f32(inv_scale);

  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  for (i = 0; i + 8 <= N; i += 8) {
    const float32x4_t vin0123 = vld1q_f32(in);
    in += 4;
    const float32x4_t vin4567 = vld1q_f32(in);
    in += 4;
    const int32x4_t v0123_rounded =
        vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
    const int32x4_t v4567_rounded =
        vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));
    const int16x8_t v01234567_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded), vzero_point);
    const underlying_x8_t vout01234567 =
        vqmov<underlying_x8_t>(v01234567_packed);
    vst1<underlying_t, underlying_x8_t>(out_underlying, vout01234567);
    out_underlying += 8;
  }
  for (; i < N; ++i) {
    (*out_underlying++) =
        quantize_val<underlying_t>(scale, zero_point, (*in++));
  }
}

template <typename T>
void quantize_tensor_arm64_q8_wrapper(
    const float* __restrict__ in,
    T* __restrict__ out,
    const int64_t N,
    const float scale,
    const int32_t zero_point);

#endif /* __aarch64__ */

template <typename T = uint8_t>
executorch::runtime::Error QuantizePerTensor(
    const executorch::aten::Tensor& rtensor,
    executorch::aten::Tensor& qtensor,
    double scale,
    int zero_point) {
  const float* rdata = rtensor.const_data_ptr<float>();
  int numel = rtensor.numel();
  ET_CHECK_OR_RETURN_ERROR(
      (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value),
      Internal,
      "Expecting quantized output tensor of dtype uint8_t or int8_t");
  ET_CHECK_OR_RETURN_ERROR(
      rtensor.numel() <= qtensor.numel(),
      Internal,
      "Expecting quantized output tensor of same or smaller size as input, %zd vs. %zd",
      qtensor.numel(),
      rtensor.numel());
  T* qdata = qtensor.mutable_data_ptr<T>();

#if defined(__aarch64__)
  quantize_tensor_arm64_q8_wrapper<T>(rdata, qdata, numel, scale, zero_point);
#else
  for (int i = 0; i < numel; ++i) {
    qdata[i] = quantize_val<T>(scale, zero_point, rdata[i]);
  }
#endif /* __aarch64__ */
  return executorch::runtime::Error::Ok;
}

executorch::runtime::Error GenerateRequantizationScale(
    const executorch::aten::Tensor& weight_scales,
    float input_scale,
    float output_scale,
    std::vector<float>& requant_scales);

std::pair<float, float> GetMinMax(const executorch::aten::Tensor& ft);

} // namespace utils
} // namespace xnnpack
} // namespace backends
} // namespace executorch
