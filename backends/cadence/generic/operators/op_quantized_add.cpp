/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_add.h>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/quantized_op_macros.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl::generic::native {

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::dequantize;
using ::impl::generic::kernels::quantize;

DECLARE_POINTWISE_TENSOR_QUANTIZED_BINARY_OP(quantized_add_, +);

#define DECLARE_POINTWISE_SCALAR_QUANTIZED_BINARY_OP(BINARY_FUNC_NAME, OP) \
  template <typename T>                                                    \
  void BINARY_FUNC_NAME(                                                   \
      const Tensor& X,                                                     \
      float X_scale,                                                       \
      int32_t X_zero_point,                                                \
      const float Y,                                                       \
      float out_scale,                                                     \
      int32_t out_zero_point,                                              \
      Tensor& out) {                                                       \
    const T* __restrict__ X_data = X.const_data_ptr<T>();                  \
    T* __restrict__ out_data = out.mutable_data_ptr<T>();                  \
    float inv_out_scale = 1.0f / out_scale;                                \
    for (size_t i = 0, e = X.numel(); i < e; ++i) {                        \
      float x = dequantize<T>(X_data[i], X_scale, X_zero_point);           \
      float z = x OP Y;                                                    \
      out_data[i] = quantize<T>(z, inv_out_scale, out_zero_point);         \
    }                                                                      \
  }

DECLARE_POINTWISE_SCALAR_QUANTIZED_BINARY_OP(quantized_add_Scalar_, +);

Tensor& quantized_add_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    const Tensor& X_scale_t,
    const Tensor& X_zero_point_t,
    const Tensor& Y,
    const Tensor& Y_scale_t,
    const Tensor& Y_zero_point_t,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  float X_scale = X_scale_t.const_data_ptr<float>()[0];
  int32_t X_zero_point = X_zero_point_t.const_data_ptr<int32_t>()[0];
  float Y_scale = Y_scale_t.const_data_ptr<float>()[0];
  int32_t Y_zero_point = Y_zero_point_t.const_data_ptr<int32_t>()[0];

#define typed_quantized_add(ctype, dtype)     \
  case ScalarType::dtype: {                   \
    quantized_add_<ctype>(                    \
        X,                                    \
        X_scale,                              \
        X_zero_point,                         \
        Y,                                    \
        Y_scale,                              \
        Y_zero_point,                         \
        static_cast<float>(out_scale),        \
        static_cast<int32_t>(out_zero_point), \
        out);                                 \
    break;                                    \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_add);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
#undef typed_quantized_add

  return out;
}

Tensor& quantized_add_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
#define typed_quantized_add(ctype, dtype)     \
  case ScalarType::dtype: {                   \
    quantized_add_<ctype>(                    \
        X,                                    \
        static_cast<float>(X_scale),          \
        static_cast<int32_t>(X_zero_point),   \
        Y,                                    \
        static_cast<float>(Y_scale),          \
        static_cast<int32_t>(Y_zero_point),   \
        static_cast<float>(out_scale),        \
        static_cast<int32_t>(out_zero_point), \
        out);                                 \
    break;                                    \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_add);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
#undef typed_quantized_add
  return out;
}

Tensor& quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  quantized_add_<int8_t>(
      X,
      static_cast<float>(X_scale),
      static_cast<int32_t>(X_zero_point),
      Y,
      static_cast<float>(Y_scale),
      static_cast<int32_t>(Y_zero_point),
      static_cast<float>(out_scale),
      static_cast<int32_t>(out_zero_point),
      out);
  return out;
}

Tensor& quantized_add_asym8uxasym8u_asym8u_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  quantized_add_<uint8_t>(
      X,
      static_cast<float>(X_scale),
      static_cast<int32_t>(X_zero_point),
      Y,
      static_cast<float>(Y_scale),
      static_cast<int32_t>(Y_zero_point),
      static_cast<float>(out_scale),
      static_cast<int32_t>(out_zero_point),
      out);
  return out;
}

Tensor& quantized_add_Scalar_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    const Tensor& X_scale_t,
    const Tensor& X_zero_point_t,
    const Scalar& Y_scalar,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  float X_scale = X_scale_t.const_data_ptr<float>()[0];
  int32_t X_zero_point = X_zero_point_t.const_data_ptr<int32_t>()[0];
  float Y = static_cast<float>(
      ::torch::executor::native::utils::scalar_to<double>(Y_scalar));
#define typed_quantized_add_Scalar(ctype, dtype) \
  case ScalarType::dtype: {                      \
    quantized_add_Scalar_<ctype>(                \
        X,                                       \
        X_scale,                                 \
        X_zero_point,                            \
        Y,                                       \
        static_cast<float>(out_scale),           \
        static_cast<int32_t>(out_zero_point),    \
        out);                                    \
    break;                                       \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_add_Scalar)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
#undef typed_quantized_add_Scalar
  return out;
}

#undef DECLARE_POINTWISE_SCALAR_QUANTIZED_BINARY_OP

} // namespace impl::generic::native
