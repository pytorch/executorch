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

/**
 * For an input tensor, use the scale and zero_point arguments to quantize it.
 */
namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;

namespace {

/**
 * Asserts that the parameters are valid.
 */
void check_quantize_per_tensor_args(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Ensure self and out has the same shape
  ET_CHECK_MSG(
      torch::executor::isFloatingType(input.scalar_type()),
      "input.scalar_type() %" PRId8 " is not floating type",
      static_cast<int8_t>(input.scalar_type()));

  int32_t quant_min_lower_bound = 0, quant_max_upper_bound = 0;
  ScalarType out_dtype = out.scalar_type();
  ET_CHECK_MSG(
      out_dtype == dtype,
      "out.scalar_type() %" PRId8 " is not matching dtype argument %" PRId8,
      static_cast<int8_t>(out_dtype),
      static_cast<int8_t>(dtype));
  if (out_dtype == ScalarType::Byte) {
    quant_min_lower_bound =
        static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
    quant_max_upper_bound =
        static_cast<int32_t>(std::numeric_limits<uint8_t>::max());
  } else if (dtype == ScalarType::Char) {
    quant_min_lower_bound =
        static_cast<int32_t>(std::numeric_limits<int8_t>::min());
    quant_max_upper_bound =
        static_cast<int32_t>(std::numeric_limits<int8_t>::max());
  } else if (dtype == ScalarType::Short) {
    quant_min_lower_bound = std::numeric_limits<int16_t>::min();
    quant_max_upper_bound = std::numeric_limits<int16_t>::max();
  } else if (dtype == ScalarType::Int) {
    quant_min_lower_bound = std::numeric_limits<int32_t>::min();
    quant_max_upper_bound = std::numeric_limits<int32_t>::max();
  } else {
    ET_CHECK_MSG(
        false, "Unsupported dtype: %" PRId8, static_cast<int8_t>(out_dtype));
  }
  ET_CHECK_MSG(
      quant_min >= quant_min_lower_bound,
      "quant_min out of bound for dtype, expected quant_min_lower_bound: %" PRId32
      " actual quant_min: %" PRId64,
      quant_min_lower_bound,
      quant_min);

  ET_CHECK_MSG(
      quant_max <= quant_max_upper_bound,
      "quant_max out of bound for dtype, expected quant_max_upper_bound: %" PRId32
      " actual quant_max: %" PRId64,
      quant_max_upper_bound,
      quant_max);
}

} // namespace

template <typename T, typename K>
T quantize_val(
    double scale,
    int64_t zero_point,
    K value,
    int64_t quant_min,
    int64_t quant_max) {
  int64_t qvalue;
  float inv_scale = 1.0f / static_cast<float>(scale);
  qvalue = static_cast<int64_t>(
      static_cast<int32_t>(zero_point) +
      std::nearbyint(static_cast<float>(inv_scale * value)));

  qvalue = std::max<int64_t>(qvalue, quant_min);
  qvalue = std::min<int64_t>(qvalue, quant_max);
  return static_cast<T>(qvalue);
}

Tensor& quantize_per_tensor_out(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in quantize_per_tensor_out");

  check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out);

  // calculate the quantized input
#define QUANTIZE_IMPL(IN_CTYPE, OUT_CTYPE, out_dtype)                   \
  case ScalarType::out_dtype:                                           \
    for (size_t i = 0; i < input.numel(); i++) {                        \
      IN_CTYPE value = input.data_ptr<IN_CTYPE>()[i];                   \
      out.data_ptr<OUT_CTYPE>()[i] = quantize_val<OUT_CTYPE, IN_CTYPE>( \
          scale, zero_point, value, quant_min, quant_max);              \
    }                                                                   \
    break;
#define CALCULATE_FLOAT_TYPE(IN_CTYPE, in_dtype)         \
  case ScalarType::in_dtype:                             \
    switch (out.scalar_type()) {                         \
      ET_FORALL_INT_TYPES_WITH(IN_CTYPE, QUANTIZE_IMPL); \
      default:                                           \
        ET_CHECK_MSG(                                    \
            false,                                       \
            "Unhandled output dtype %" PRId8,            \
            static_cast<int8_t>(out.scalar_type()));     \
    }                                                    \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_FLOAT_TYPES(CALCULATE_FLOAT_TYPE);
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(input.scalar_type()));
  }
#undef CALCULATE_FLOAT_TYPE
#undef QUANTIZE_IMPL
  return out;
}

Tensor& quantize_per_tensor_tensor_args_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  ET_CHECK_MSG(
      scale.scalar_type() == ScalarType::Double,
      "Expected scale to be Double tensor received: %" PRId8,
      static_cast<int8_t>(scale.scalar_type()));
  ET_CHECK_MSG(
      zero_point.scalar_type() == ScalarType::Long,
      "Expected zero_point to be Long tensor received: %" PRId8,
      static_cast<int8_t>(zero_point.scalar_type()));
  ET_CHECK_MSG(
      scale.numel() == 1,
      "Exepcted scale to only have one element received: %zd",
      ssize_t(scale.numel()));
  ET_CHECK_MSG(
      zero_point.numel() == 1,
      "Exepcted zero_point to only have one element received: %zd",
      ssize_t(zero_point.numel()));

  quantize_per_tensor_out(
      input,
      scale.data_ptr<double>()[0],
      zero_point.data_ptr<int64_t>()[0],
      quant_min,
      quant_max,
      dtype,
      out);
  return out;
}

Tensor& quantize_per_tensor_out(
    RuntimeContext& context,

    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}

Tensor& quantize_per_tensor_tensor_args_out(
    RuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)context;
  return quantize_per_tensor_tensor_args_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}

} // namespace native
} // namespace executor
} // namespace torch
