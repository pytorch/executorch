/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cinttypes>
#include <cmath>

// Check for Helium/MVE support
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#define HAS_HELIUM_SIMD 1
#endif

namespace cortex_m {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

/**
 * Asserts that the parameters are valid for float to int8 quantization.
 */
void check_quantize_args(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Ensure input is float type
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Float,
      "input.scalar_type() %" PRId8 " is not float type",
      static_cast<int8_t>(input.scalar_type()));

  // Check output dtype is int8
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Char,
      "out.scalar_type() %" PRId8 " is not int8 (Char)",
      static_cast<int8_t>(out.scalar_type()));

  // Check dtype is int8
  ET_CHECK_MSG(
      dtype == ScalarType::Char,
      "dtype %" PRId8 " is not int8 (Char)",
      static_cast<int8_t>(dtype));

  // Validate quant_min and quant_max for int8
  int32_t quant_min_lower_bound = std::numeric_limits<int8_t>::min();
  int32_t quant_max_upper_bound = std::numeric_limits<int8_t>::max();

  ET_CHECK_MSG(
      quant_min >= quant_min_lower_bound,
      "quant_min out of bound for int8, expected quant_min_lower_bound: %" PRId32
      " actual quant_min: %" PRId64,
      quant_min_lower_bound,
      quant_min);

  ET_CHECK_MSG(
      quant_max <= quant_max_upper_bound,
      "quant_max out of bound for int8, expected quant_max_upper_bound: %" PRId32
      " actual quant_max: %" PRId64,
      quant_max_upper_bound,
      quant_max);
}

/**
 * Scalar implementation of quantization for a single value.
 */
template <typename Q, typename F>
Q quantize_val(
    F inv_scale,
    int32_t zero_point,
    F value,
    int64_t quant_min,
    int64_t quant_max) {
  int32_t qvalue =
      zero_point + static_cast<int32_t>(std::nearbyint(inv_scale * value));
  qvalue = std::max<int32_t>(qvalue, static_cast<int32_t>(quant_min));
  qvalue = std::min<int32_t>(qvalue, static_cast<int32_t>(quant_max));
  return static_cast<Q>(qvalue);
}

} // namespace

Tensor& quantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Ignore context for now
  (void)context;

  // Resize output tensor to match input dimensions
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in quantize_per_tensor_out");

  // Validate input parameters
  check_quantize_args(input, quant_min, quant_max, dtype, out);

  // Pre-compute inverse scale for better performance
  float inv_scale = 1.0f / static_cast<float>(scale);
  int32_t zp = static_cast<int32_t>(zero_point);
  int32_t qmin = static_cast<int32_t>(quant_min);
  int32_t qmax = static_cast<int32_t>(quant_max);

  // Get pointers to input and output data
  const float* input_data = input.const_data_ptr<float>();
  int8_t* out_data = out.mutable_data_ptr<int8_t>();
  const size_t numel = input.numel();

  size_t i = 0;

#if defined(HAS_HELIUM_SIMD)
  // Helium MVE implementation for float32 to int8 quantization
  static uint8x16_t voffset{
      0x0,
      0x8,
      0x4,
      0xC,
      0x1,
      0x9,
      0x5,
      0xD,
      0x2,
      0xA,
      0x6,
      0xE,
      0x3,
      0xB,
      0x7,
      0xF};

  float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);

  // Magic number for float to int conversion, round to nearest even integer
  // int magic_round(float f): interpret_as_int32(f + magic_float) - magic_int
  // where,
  //    magic_float = 12582912.0f = (2 ** 23 + 2 ** 22) = (1.5 * 2 ** 23)
  //    magic_int = 1262485504 = 0x4B400000 = bit_pattern_as_int32(magic_float)

  float magic_float = 12582912.0f;
  int32_t magic_int = 1262485504;

  float32x4_t vmagic_float = vdupq_n_f32(magic_float);
  int32x4_t vmagic_int_less_zp =
      vdupq_n_s32(magic_int - static_cast<int32_t>(zp));

  int16x8_t vqmin = vdupq_n_s16(qmin);
  int16x8_t vqmax = vdupq_n_s16(qmax);

  // TODO: Measure performnce, we are spilling
  for (; i + 15 < numel; i += 16) {
    float32x4_t in_0123 = vldrwq_f32(input_data + 0);
    float32x4_t in_4567 = vldrwq_f32(input_data + 4);
    float32x4_t in_89AB = vldrwq_f32(input_data + 8);
    float32x4_t in_CDEF = vldrwq_f32(input_data + 12);

    float32x4_t outf_0123 = vfmaq_f32(vmagic_float, in_0123, inv_scale_vec);
    float32x4_t outf_4567 = vfmaq_f32(vmagic_float, in_4567, inv_scale_vec);
    float32x4_t outf_89AB = vfmaq_f32(vmagic_float, in_89AB, inv_scale_vec);
    float32x4_t outf_CDEF = vfmaq_f32(vmagic_float, in_CDEF, inv_scale_vec);

    int32x4_t out_0123 =
        vsubq_s32(vreinterpretq_s32_f32(outf_0123), vmagic_int_less_zp);
    int32x4_t out_4567 =
        vsubq_s32(vreinterpretq_s32_f32(outf_4567), vmagic_int_less_zp);
    int32x4_t out_89AB =
        vsubq_s32(vreinterpretq_s32_f32(outf_89AB), vmagic_int_less_zp);
    int32x4_t out_CDEF =
        vsubq_s32(vreinterpretq_s32_f32(outf_CDEF), vmagic_int_less_zp);

    int16x8_t out_04152637;
    int16x8_t out_8C9DAEBF;
    out_04152637 = vmovnbq_s32(out_04152637, out_0123);
    out_04152637 = vmovntq_s32(out_04152637, out_4567);
    out_8C9DAEBF = vmovnbq_s32(out_8C9DAEBF, out_89AB);
    out_8C9DAEBF = vmovntq_s32(out_8C9DAEBF, out_CDEF);

    int16x8_t out_04152637_clamped =
        vminq_s16(vmaxq_s16(out_04152637, vqmin), vqmax);
    int16x8_t out_8C9DAEBF_clamped =
        vminq_s16(vmaxq_s16(out_8C9DAEBF, vqmin), vqmax);

    int8x16_t out_084C195D2A6E3B7F;
    out_084C195D2A6E3B7F =
        vmovnbq_s16(out_084C195D2A6E3B7F, out_04152637_clamped);
    out_084C195D2A6E3B7F =
        vmovntq_s16(out_084C195D2A6E3B7F, out_8C9DAEBF_clamped);

    vstrbq_scatter_offset_s8(out_data, voffset, out_084C195D2A6E3B7F);
    input_data += 16;
    out_data += 16;
  }
#endif // defined(HAS_HELIUM_SIMD)

  for (; i < numel; i++) {
    *out_data =
        quantize_val<int8_t, float>(inv_scale, zp, *input_data, qmin, qmax);
    input_data++;
    out_data++;
  }

  return out;
}

Tensor& quantize_per_tensor_out(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  KernelRuntimeContext context;
  return quantize_per_tensor_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
}

} // namespace native
} // namespace cortex_m
