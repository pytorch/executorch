/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// Requantize the int8_t/uint8_t input tensor to a uint8_t/int8_t out tensor.
// The scale and zero_point for requantization are in the args.
Tensor& requantize_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_scale_t,
    const Tensor& in_zero_point_t,
    const Tensor& out_scale_t,
    const Tensor& out_zero_point_t,
    const ScalarType out_dtype,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      in_scale_t.scalar_type() == ScalarType::Float,
      InvalidArgument,
      out,
      "In scale is not a float: %s",
      torch::executor::toString(in_scale_t.scalar_type()));
  float in_scale = in_scale_t.const_data_ptr<float>()[0];

  ET_KERNEL_CHECK_MSG(
      ctx,
      in_zero_point_t.scalar_type() == ScalarType::Int,
      InvalidArgument,
      out,
      "In zero point is not an int: %s",
      torch::executor::toString(in_zero_point_t.scalar_type()));
  int32_t in_zero_point = in_zero_point_t.const_data_ptr<int32_t>()[0];

  ET_KERNEL_CHECK_MSG(
      ctx,
      out_scale_t.scalar_type() == ScalarType::Float,
      InvalidArgument,
      out,
      "Out scale is not a float: %s",
      torch::executor::toString(out_scale_t.scalar_type()));
  float out_scale = out_scale_t.const_data_ptr<float>()[0];

  ET_KERNEL_CHECK_MSG(
      ctx,
      out_zero_point_t.scalar_type() == ScalarType::Int,
      InvalidArgument,
      out,
      "Out zero point is not an int: %s",
      torch::executor::toString(out_zero_point_t.scalar_type()));
  int32_t out_zero_point = out_zero_point_t.const_data_ptr<int32_t>()[0];

  ET_KERNEL_CHECK_MSG(
      ctx,
      out.scalar_type() == out_dtype,
      InvalidArgument,
      out,
      "Out tensor dtype (%s) does not match the passed in out dtype (%s)",
      torch::executor::toString(out.scalar_type()),
      torch::executor::toString(out_dtype));

  const size_t numel = out.numel();
  ScalarType in_dtype = input.scalar_type();

  // Assert that the output tensor's dtype is same as out_dtype.
  ET_KERNEL_CHECK_MSG(
      ctx,
      out_dtype == out.scalar_type(),
      InvalidArgument,
      out,
      "Out dtype %s does not match requant dtype %s",
      torch::executor::toString(out.scalar_type()),
      torch::executor::toString(out_dtype));

#define typed_requantize(ctype, dtype)                                      \
  const ctype* input_data = input.const_data_ptr<ctype>();                  \
  dtype* out_data = out.mutable_data_ptr<dtype>();                          \
  for (size_t i = 0; i < numel; ++i) {                                      \
    float dequant =                                                         \
        kernels::dequantize<ctype>(input_data[i], in_scale, in_zero_point); \
    out_data[i] =                                                           \
        kernels::quantize<dtype>(dequant, 1 / out_scale, out_zero_point);   \
  };
#define typed_requantize_in(ctype)               \
  switch (out_dtype) {                           \
    case ScalarType::Byte: {                     \
      typed_requantize(ctype, uint8_t);          \
      break;                                     \
    }                                            \
    case ScalarType::Char: {                     \
      typed_requantize(ctype, int8_t);           \
      break;                                     \
    }                                            \
    case ScalarType::UInt16: {                   \
      typed_requantize(ctype, uint16_t);         \
      break;                                     \
    }                                            \
    case ScalarType::Short: {                    \
      typed_requantize(ctype, int16_t);          \
      break;                                     \
    }                                            \
    default:                                     \
      ET_KERNEL_CHECK_MSG(                       \
          ctx,                                   \
          false,                                 \
          InvalidArgument,                       \
          out,                                   \
          "Unhandled output dtype %s",           \
          torch::executor::toString(out_dtype)); \
  }

  switch (in_dtype) {
    case ScalarType::Byte: {
      typed_requantize_in(uint8_t);
      break;
    }
    case ScalarType::Char: {
      typed_requantize_in(int8_t);
      break;
    }
    case ScalarType::UInt16: {
      typed_requantize_in(uint16_t);
      break;
    }
    case ScalarType::Short: {
      typed_requantize_in(int16_t);
      break;
    }
    default:
      ET_KERNEL_CHECK_MSG(
          ctx,
          false,
          InvalidArgument,
          out,
          "Unhandled input dtype %s",
          torch::executor::toString(in_dtype));
  }
#undef typed_requantize_in
#undef typed_requantize
  return out;
}

// Requantize the int8_t/uint8_t input tensor to a uint8_t/int8_t out tensor.
// The scale and zero_point for requantization are in the args.
Tensor& requantize_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double in_scale,
    int64_t in_zero_point,
    double out_scale,
    int64_t out_zero_point,
    const ScalarType out_dtype,
    Tensor& out) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      out.scalar_type() == out_dtype,
      InvalidArgument,
      out,
      "Out tensor dtype (%s) does not match the passed in out dtype (%s)",
      torch::executor::toString(out.scalar_type()),
      torch::executor::toString(out_dtype));

  const size_t numel = out.numel();
  ScalarType in_dtype = input.scalar_type();

  // Assert that the output tensor's dtype is same as out_dtype.
  ET_KERNEL_CHECK_MSG(
      ctx,
      out_dtype == out.scalar_type(),
      InvalidArgument,
      out,
      "Out dtype %s does not match requant dtype %s",
      torch::executor::toString(out.scalar_type()),
      torch::executor::toString(out_dtype));

#define typed_requantize(ctype, dtype)                                      \
  const ctype* input_data = input.const_data_ptr<ctype>();                  \
  dtype* out_data = out.mutable_data_ptr<dtype>();                          \
  for (size_t i = 0; i < numel; ++i) {                                      \
    float dequant =                                                         \
        kernels::dequantize<ctype>(input_data[i], in_scale, in_zero_point); \
    out_data[i] =                                                           \
        kernels::quantize<dtype>(dequant, 1 / out_scale, out_zero_point);   \
  };

#define typed_requantize_in(ctype)               \
  switch (out_dtype) {                           \
    case ScalarType::Byte: {                     \
      typed_requantize(ctype, uint8_t);          \
      break;                                     \
    }                                            \
    case ScalarType::Char: {                     \
      typed_requantize(ctype, int8_t);           \
      break;                                     \
    }                                            \
    case ScalarType::UInt16: {                   \
      typed_requantize(ctype, uint16_t);         \
      break;                                     \
    }                                            \
    case ScalarType::Short: {                    \
      typed_requantize(ctype, int16_t);          \
      break;                                     \
    }                                            \
    default:                                     \
      ET_KERNEL_CHECK_MSG(                       \
          ctx,                                   \
          false,                                 \
          InvalidArgument,                       \
          out,                                   \
          "Unhandled output dtype %s",           \
          torch::executor::toString(out_dtype)); \
  }

  switch (in_dtype) {
    case ScalarType::Byte: {
      typed_requantize_in(uint8_t);
      break;
    }
    case ScalarType::Char: {
      typed_requantize_in(int8_t);
      break;
    }
    case ScalarType::UInt16: {
      typed_requantize_in(uint16_t);
      break;
    }
    case ScalarType::Short: {
      typed_requantize_in(int16_t);
      break;
    }
    default:
      ET_KERNEL_CHECK_MSG(
          ctx,
          false,
          InvalidArgument,
          out,
          "Unhandled input dtype %s",
          torch::executor::toString(in_dtype));
  }
#undef typed_requantize_in
#undef typed_requantize
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
