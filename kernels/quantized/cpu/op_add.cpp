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

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

namespace {

template <typename INPUT_T, typename OUTPUT_T>
OUTPUT_T quantize_val(
    double scale,
    int64_t zero_point,
    INPUT_T value,
    int64_t quant_min,
    int64_t quant_max) {
  int64_t qvalue;
  float inv_scale = 1.0f / static_cast<float>(scale);
  qvalue = static_cast<int64_t>(zero_point + std::nearbyint(inv_scale * value));
  qvalue = std::max<int64_t>(qvalue, quant_min);
  qvalue = std::min<int64_t>(qvalue, quant_max);
  return static_cast<OUTPUT_T>(qvalue);
}

template <typename INPUT_T, typename OUTPUT_T>
OUTPUT_T dequantize_val(double scale, int64_t zero_point, INPUT_T value) {
  return (value - zero_point) * scale;
}

/**
 * Perform element wise addition of the input tensors into out.
 * Should be numerically equivalent to Dq -> fp add -> Q
 */
template <class CTYPE>
void add_tensors(
    const Tensor& a,
    float a_scale,
    int32_t a_zero_point,
    const Tensor& b,
    float b_scale,
    int32_t b_zero_point,
    Tensor& out,
    float out_scale,
    int32_t out_zero_point,
    int64_t out_quant_min,
    int64_t out_quant_max) {
  const size_t n = a.numel();

  const auto data_a = a.const_data_ptr<CTYPE>();
  const auto data_b = b.const_data_ptr<CTYPE>();
  auto data_out = out.mutable_data_ptr<CTYPE>();

  for (size_t i = 0; i < n; ++i) {
    // Dq -> fp add -> Q. Can be optimized further
    const auto dqa =
        dequantize_val<CTYPE, float>(a_scale, a_zero_point, data_a[i]);
    const auto dqb =
        dequantize_val<CTYPE, float>(b_scale, b_zero_point, data_b[i]);
    const auto accumulate = dqa + dqb;

    data_out[i] = quantize_val<float, CTYPE>(
        out_scale, out_zero_point, accumulate, out_quant_min, out_quant_max);
  }
}

} // namespace

/**
 * Perform element wise addition of the input tensors into out. Should be
 * numerically equivalent to Dq -> fp add -> Q
 *
 * PREREQ: a and b should be the same shape, quant_min and max should be in
 * range [0,255]. a and b and out should be the same dtype.
 */
Tensor& quantized_add_out(
    const Tensor& a,
    double a_scale_d,
    int64_t a_zero_point_l,
    int64_t a_quant_min,
    int64_t a_quant_max,
    const Tensor& b,
    double b_scale_d,
    int64_t b_zero_point_l,
    int64_t b_quant_min,
    int64_t b_quant_max,
    double out_scale_d,
    int64_t out_zero_point_l,
    int64_t out_quant_min,
    int64_t out_quant_max,
    Tensor& out) {
  ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, out);
  ET_CHECK_MSG(
      a_quant_min >= 0 && a_quant_max <= 255 && a_quant_min <= a_quant_max,
      "invalid quant_min: %" PRId64 " or quant_max: %" PRId64
      " for input tensor a. Min should be <= max and both should be in bounds [0,255]",
      a_quant_min,
      a_quant_max);
  ET_CHECK_MSG(
      b_quant_min >= 0 && b_quant_max <= 255 && b_quant_min <= b_quant_max,
      "invalid quant_min: %" PRId64 " or quant_max: %" PRId64
      " for input tensor b. Min should be <= max and both should be in bounds [0,255]",
      b_quant_min,
      b_quant_max);
  ET_CHECK_MSG(
      out_quant_min >= 0 && out_quant_max <= 255 &&
          out_quant_min <= out_quant_max,
      "invalid quant_min: %" PRId64 " or quant_max: %" PRId64
      " for output tensor. Min should be <= max and both should be in bounds [0,255]",
      out_quant_min,
      out_quant_max);

  // downsize to maintain numerical consistency with fbgemm
  float a_scale = static_cast<float>(a_scale_d);
  float b_scale = static_cast<float>(b_scale_d);
  float out_scale = static_cast<float>(out_scale_d);

  int32_t a_zero_point = static_cast<int32_t>(a_zero_point_l);
  int32_t b_zero_point = static_cast<int32_t>(b_zero_point_l);
  int32_t out_zero_point = static_cast<int32_t>(out_zero_point_l);

#define ADD_TENSORS(ctype, dtype) \
  case ScalarType::dtype:         \
    add_tensors<ctype>(           \
        a,                        \
        a_scale,                  \
        a_zero_point,             \
        b,                        \
        b_scale,                  \
        b_zero_point,             \
        out,                      \
        out_scale,                \
        out_zero_point,           \
        out_quant_min,            \
        out_quant_max);           \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_INT_TYPES(ADD_TENSORS)
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled dtype %" PRId8,
          static_cast<int8_t>(a.scalar_type()));
  }

#undef ADD_TENSORS

  return out;
}

Tensor& quantized_add_out(
    KernelRuntimeContext& context,
    const Tensor& a,
    double a_scale,
    int64_t a_zero_point,
    int64_t a_quant_min,
    int64_t a_quant_max,
    const Tensor& b,
    double b_scale,
    int64_t b_zero_point,
    int64_t b_quant_min,
    int64_t b_quant_max,
    double out_scale,
    int64_t out_zero_point,
    int64_t out_quant_min,
    int64_t out_quant_max,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return quantized_add_out(
      a,
      a_scale,
      a_zero_point,
      a_quant_min,
      a_quant_max,
      b,
      b_scale,
      b_zero_point,
      b_quant_min,
      b_quant_max,
      out_scale,
      out_zero_point,
      out_quant_min,
      out_quant_max,
      out);
}

} // namespace native
} // namespace executor
} // namespace torch
