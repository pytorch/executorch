/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/operators.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::dequantize;
using ::impl::generic::kernels::quantize;

template <typename T>
void quantized_add_per_tensor_impl(
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  const T* __restrict__ X_data = X.const_data_ptr<T>();
  const T* __restrict__ Y_data = Y.const_data_ptr<T>();
  T* __restrict__ out_data = out.mutable_data_ptr<T>();

  ssize_t Y_numel = Y.numel();
  ssize_t X_numel = X.numel();
  ssize_t out_numel = out.numel();

  float X_scale_f = static_cast<float>(X_scale);
  float Y_scale_f = static_cast<float>(Y_scale);
  float out_scale_f = static_cast<float>(out_scale);
  int32_t X_zero_point_i32 = static_cast<int32_t>(X_zero_point);
  int32_t Y_zero_point_i32 = static_cast<int32_t>(Y_zero_point);
  int32_t out_zero_point_i32 = static_cast<int32_t>(out_zero_point);

  float inv_out_scale = 1.0f / out_scale_f;

  // Simple case: tensors have the same shape, no broadcasting
  if (X_numel == Y_numel && Y_numel == out_numel) {
    for (size_t i = 0; i < X_numel; ++i) {
      float x = dequantize<T>(X_data[i], X_scale_f, X_zero_point_i32);
      float y = dequantize<T>(Y_data[i], Y_scale_f, Y_zero_point_i32);
      float z = x + y;
      out_data[i] = quantize<T>(z, inv_out_scale, out_zero_point_i32);
    }
  }
  // Y is a scalar tensor
  else if (Y_numel == 1) {
    float y = dequantize<T>(Y_data[0], Y_scale_f, Y_zero_point_i32);
    for (size_t i = 0; i < X_numel; ++i) {
      float x = dequantize<T>(X_data[i], X_scale_f, X_zero_point_i32);
      float z = x + y;
      out_data[i] = quantize<T>(z, inv_out_scale, out_zero_point_i32);
    }
  }
  // X is a scalar tensor
  else if (X_numel == 1) {
    float x = dequantize<T>(X_data[0], X_scale_f, X_zero_point_i32);
    for (size_t i = 0; i < Y_numel; ++i) {
      float y = dequantize<T>(Y_data[i], Y_scale_f, Y_zero_point_i32);
      float z = x + y;
      out_data[i] = quantize<T>(z, inv_out_scale, out_zero_point_i32);
    }
  }
  // General broadcasting case - simplified implementation
  else {
    for (ssize_t i = 0; i < out_numel; ++i) {
      // Simple broadcasting: repeat elements as needed
      size_t x_idx = (X_numel == 1) ? 0 : i % X_numel;
      size_t y_idx = (Y_numel == 1) ? 0 : i % Y_numel;

      float x = dequantize<T>(X_data[x_idx], X_scale_f, X_zero_point_i32);
      float y = dequantize<T>(Y_data[y_idx], Y_scale_f, Y_zero_point_i32);
      float z = x + y;
      out_data[i] = quantize<T>(z, inv_out_scale, out_zero_point_i32);
    }
  }
}

// Generic quantized add with type dispatch
void quantized_add_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  (void)ctx;

  executorch::aten::ScalarType dtype = X.scalar_type();
  switch (dtype) {
    case executorch::aten::ScalarType::Byte:
      quantized_add_per_tensor_impl<uint8_t>(
          X,
          X_scale,
          X_zero_point,
          Y,
          Y_scale,
          Y_zero_point,
          out_scale,
          out_zero_point,
          out);
      break;
    case executorch::aten::ScalarType::Char:
      quantized_add_per_tensor_impl<int8_t>(
          X,
          X_scale,
          X_zero_point,
          Y,
          Y_scale,
          Y_zero_point,
          out_scale,
          out_zero_point,
          out);
      break;
    default:
      ET_CHECK_MSG(
          false, "Unhandled input dtype %hhd", static_cast<int8_t>(dtype));
  }
}

// int8-specific quantized add
void quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  (void)ctx;

  quantized_add_per_tensor_impl<int8_t>(
      X,
      X_scale,
      X_zero_point,
      Y,
      Y_scale,
      Y_zero_point,
      out_scale,
      out_zero_point,
      out);
}

// uint8-specific quantized add
void quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  (void)ctx;

  quantized_add_per_tensor_impl<uint8_t>(
      X,
      X_scale,
      X_zero_point,
      Y,
      Y_scale,
      Y_zero_point,
      out_scale,
      out_zero_point,
      out);
}

} // namespace native
} // namespace generic
} // namespace impl
