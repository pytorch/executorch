/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cinttypes>

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
void check_dequantize_args(
    const Tensor& input,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Ensure input is char type
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Char,
      "input.scalar_type() %" PRId8 " is not char type",
      static_cast<int8_t>(input.scalar_type()));

  // Check zp range
  ET_CHECK_MSG(
      zero_point >= quant_min,
      "zero_point must be %" PRId64 " <= quant_min %" PRId64,
      zero_point,
      quant_min);
  ET_CHECK_MSG(
      zero_point <= quant_max,
      "zero_point must be %" PRId64 " >= quant_max %" PRId64,
      zero_point,
      quant_max);

  // Check output dtype is float
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Float,
      "out.scalar_type() %" PRId8 " is not float",
      static_cast<int8_t>(out.scalar_type()));

  // Check dtype is int8 (Char)
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
F dequantize_val(float scale, int32_t zero_point, Q qvalue) {
  return static_cast<F>((static_cast<int32_t>(qvalue) - zero_point) * scale);
}
} // namespace

Tensor& dequantize_per_tensor_out(
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
      "Failed to resize out Tensor in dequantize_per_tensor_out");

  // Validate input parameters
  check_dequantize_args(input, zero_point, quant_min, quant_max, dtype, out);

  int32_t zp = static_cast<int32_t>(zero_point);

  // Get pointers to input and output data
  const int8_t* input_data = input.const_data_ptr<int8_t>();
  float* out_data = out.mutable_data_ptr<float>();
  const size_t numel = input.numel();

  size_t i = 0;
#if defined(HAS_HELIUM_SIMD)
  // Helium MVE implementation for int8 to float quantization
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

  int16x8_t vzp = vdupq_n_s16(static_cast<int16_t>(zp));
  float32x4_t vscale = vdupq_n_f32(static_cast<float>(scale));

  for (; i + 15 < numel; i += 16) {
    int8x16_t in_084C195D2A6E3B7F =
        vldrbq_gather_offset_s8(input_data, voffset);

    int16x8_t in_04152637 = vsubq_s16(vmovlbq_s8(in_084C195D2A6E3B7F), vzp);
    int16x8_t in_8C9DAEBF = vsubq_s16(vmovltq_s8(in_084C195D2A6E3B7F), vzp);

    float32x4_t inf_0123 = vcvtq_f32_s32(vmovlbq_s16(in_04152637));
    float32x4_t inf_4567 = vcvtq_f32_s32(vmovltq_s16(in_04152637));
    float32x4_t inf_89AB = vcvtq_f32_s32(vmovlbq_s16(in_8C9DAEBF));
    float32x4_t inf_CDEF = vcvtq_f32_s32(vmovltq_s16(in_8C9DAEBF));

    float32x4_t out_0123 = vmulq_f32(inf_0123, vscale);
    float32x4_t out_4567 = vmulq_f32(inf_4567, vscale);
    float32x4_t out_89AB = vmulq_f32(inf_89AB, vscale);
    float32x4_t out_CDEF = vmulq_f32(inf_CDEF, vscale);

    vstrwq_f32(out_data + 0, out_0123);
    vstrwq_f32(out_data + 4, out_4567);
    vstrwq_f32(out_data + 8, out_89AB);
    vstrwq_f32(out_data + 12, out_CDEF);

    input_data += 16;
    out_data += 16;
  }
#endif // defined(HAS_HELIUM_SIMD)

  for (; i < numel; i++) {
    *out_data = dequantize_val<int8_t, float>(scale, zp, *input_data);
    input_data++;
    out_data++;
  }
  return out;
}

Tensor& dequantize_per_tensor_out(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  KernelRuntimeContext context;
  return dequantize_per_tensor_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
}

} // namespace native
} // namespace cortex_m
