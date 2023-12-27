/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
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
void check_dequantize_per_tensor_args(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Byte ||
          input.scalar_type() == ScalarType::Char ||
          input.scalar_type() == ScalarType::Short ||
          input.scalar_type() == ScalarType::Int,
      "input.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(input.scalar_type()));

  ET_CHECK_MSG(
      input.scalar_type() == dtype,
      "input.scalar_type() %" PRId8 " is not matching dtype argumenta:",
      static_cast<int8_t>(input.scalar_type()));

  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Float,
      "out.scalar_type() %" PRId8 " is not supported:",
      static_cast<int8_t>(out.scalar_type()));

  ET_CHECK_MSG(
      quant_min <= quant_max,
      "quant min: %" PRId64 " is greater than quant max: %" PRId64,
      quant_min,
      quant_max);
}

} // namespace

/**
 * Dequantizes the input tensor according to the formula (input - zero_point) *
 * scale
 *
 * NOTE: quant_min and quant_max are not used in computation, but rather
 * metadata that is passed around which can be useful for pattern matching. See
 * https://github.com/pytorch/pytorch/pull/87093#discussion_r1000841181 for more
 * info.
 */
Tensor& dequantize_per_tensor_out(
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
      "Failed to resize out Tensor in dequantize_per_tensor_out");

  check_dequantize_per_tensor_args(input, quant_min, quant_max, dtype, out);

  // calculate the dequantized output, cast scale to float to match fbgemm
  // behavior
#define DEQUANTIZE_IMPL(IN_CTYPE, OUT_CTYPE, out_dtype)                        \
  case ScalarType::out_dtype:                                                  \
    for (size_t i = 0; i < input.numel(); i++) {                               \
      out.data_ptr<OUT_CTYPE>()[i] = static_cast<OUT_CTYPE>(                   \
          (input.data_ptr<IN_CTYPE>()[i] - static_cast<int32_t>(zero_point)) * \
          static_cast<float>(scale));                                          \
    }                                                                          \
    break;
#define CALCULATE_INT_TYPE(IN_CTYPE, in_dtype)               \
  case ScalarType::in_dtype:                                 \
    switch (out.scalar_type()) {                             \
      ET_FORALL_FLOAT_TYPES_WITH(IN_CTYPE, DEQUANTIZE_IMPL); \
      default:                                               \
        ET_CHECK_MSG(                                        \
            false,                                           \
            "Unhandled output dtype %" PRId8,                \
            static_cast<int8_t>(out.scalar_type()));         \
    }                                                        \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_INT_TYPES(CALCULATE_INT_TYPE);
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled input dtype %" PRId8,
          static_cast<int8_t>(input.scalar_type()));
  }

#undef CALCULATE_FLOAT_TYPE
#undef DEQUANTIZE_IMPL
  return out;
}

Tensor& dequantize_per_tensor_tensor_args_out(
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
      "Expected scale to be Long tensor received: %" PRId8,
      static_cast<int8_t>(zero_point.scalar_type()));
  ET_CHECK_MSG(
      scale.numel() == 1,
      "Exepcted scale to only have one element received: %zd",
      ssize_t(scale.numel()));
  ET_CHECK_MSG(
      zero_point.numel() == 1,
      "Exepcted zero_point to only have one element received: %zd",
      ssize_t(zero_point.numel()));

  dequantize_per_tensor_out(
      input,
      scale.data_ptr<double>()[0],
      zero_point.data_ptr<int64_t>()[0],
      quant_min,
      quant_max,
      dtype,
      out);
  return out;
}

Tensor& dequantize_per_channel_out(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  torch::executor::Error err = resize_tensor(out, input.sizes());

  // normalize axis
  ET_CHECK_MSG(
      tensor_has_dim(input, axis),
      "axis %zd is not legal it should be -input.dim() <= axis < input.dim() %zd",
      ssize_t(axis),
      ssize_t(input.dim()));

  if (axis < 0) {
    axis += nonzero_dim(input);
  }

  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in dequantize_per_channel_out");

  ET_CHECK_MSG(
      scale.scalar_type() == ScalarType::Double,
      "scale.scalar_type() %" PRId8 " is not double type",
      static_cast<int8_t>(scale.scalar_type()));

  ET_CHECK_MSG(
      scale.numel() == input.size(axis),
      "scale.numel() %zd != input.size(axis) %zd",
      ssize_t(scale.numel()),
      ssize_t(input.size(axis)));

  ET_CHECK_MSG(
      zero_point.scalar_type() == ScalarType::Long,
      "zero_point.scalar_type() %" PRId8 " is not integer type",
      static_cast<int8_t>(zero_point.scalar_type()));

  ET_CHECK_MSG(
      zero_point.numel() == input.size(axis),
      "zero_point.numel() %zd != input.size(axis) %zd",
      ssize_t(zero_point.numel()),
      ssize_t(input.size(axis)));

  check_dequantize_per_tensor_args(input, quant_min, quant_max, dtype, out);

  // a list contains all dimensions except axis
  int64_t dims[input.dim() - 1];
  for (int64_t i = 0; i < input.dim() - 1; i++) {
    if (i < axis) {
      dims[i] = i;
    } else {
      dims[i] = i - 1;
    }
  }
  const double* scale_data = scale.const_data_ptr<double>();
  const int64_t* zero_point_data = zero_point.const_data_ptr<int64_t>();

  exec_aten::optional<exec_aten::ArrayRef<int64_t>> optional_dim_list{
      exec_aten::ArrayRef<int64_t>{dims, size_t(input.dim() - 1)}};

  // Actual dequantization logic
  // input, out are the input and output tensors
  // channel_ix is the index along the axis dimension. 0 <= channel_ix <
  // input.size(axis).
  //   i.e. if the tensor has shape (N,C,H,W), axis being 1, then channel_ix
  //   will be 0, 1, 2, ... C-1
  // in_ix is the flat index of the element you are dequantizing.
  //   in other words you are dequantizing in_data[in_ix]
#define DEQUANTIZE_IMPL(CTYPE_IN, CTYPE_OUT, out_dtype)                        \
  case ScalarType::out_dtype:                                                  \
    for (size_t channel_ix = 0; channel_ix < input.size(axis); ++channel_ix) { \
      double _scale = scale_data[channel_ix];                                  \
      int64_t _zero_point = zero_point_data[channel_ix];                       \
      apply_over_dim_list(                                                     \
          [input, out, _scale, _zero_point](size_t in_ix) {                    \
            out.mutable_data_ptr<CTYPE_OUT>()[in_ix] = static_cast<CTYPE_OUT>( \
                (input.const_data_ptr<CTYPE_IN>()[in_ix] - _zero_point) *      \
                _scale);                                                       \
          },                                                                   \
          input,                                                               \
          optional_dim_list,                                                   \
          channel_ix);                                                         \
    }                                                                          \
    break;
#define CALCULATE_FLOAT_TYPE(CTYPE_IN, in_dtype)             \
  case ScalarType::in_dtype:                                 \
    switch (out.scalar_type()) {                             \
      ET_FORALL_FLOAT_TYPES_WITH(CTYPE_IN, DEQUANTIZE_IMPL); \
      default:                                               \
        ET_CHECK_MSG(                                        \
            false,                                           \
            "Unhandled output dtype %" PRId8,                \
            static_cast<int8_t>(out.scalar_type()));         \
    }                                                        \
    break;

  switch (input.scalar_type()) {
    ET_FORALL_INT_TYPES(CALCULATE_FLOAT_TYPE);
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

Tensor& dequantize_per_channel_out(
    RuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  (void)context;
  return dequantize_per_channel_out(
      input, scale, zero_point, axis, quant_min, quant_max, dtype, out);
}

Tensor& dequantize_per_tensor_out(
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
  return dequantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}

Tensor& dequantize_per_tensor_tensor_args_out(
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
  return dequantize_per_tensor_tensor_args_out(
      input, scale, zero_point, quant_min, quant_max, dtype, out);
}

} // namespace native
} // namespace executor
} // namespace torch
