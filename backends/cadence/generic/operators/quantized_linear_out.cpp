/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/operators.h>
#include <executorch/backends/cadence/generic/operators/quantized_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::getLeadingDims;
using executorch::runtime::KernelRuntimeContext;

template <typename T>
void inline _typed_quantized_linear(
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    int64_t src_zero_point,
    const Tensor& weight_zero_point_t,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    Tensor& out) {
  const T* __restrict__ src_data = src.const_data_ptr<T>();
  const T* __restrict__ weight_data = weight.const_data_ptr<T>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  T* __restrict__ out_data = out.mutable_data_ptr<T>();

  int32_t weight_zero_point = weight_zero_point_t.const_data_ptr<int32_t>()[0];

  // input comes in shape [batch_size, in_dim]
  // weight comes in shape [out_dim, in_dim]
  // output comes in empty with shape [batch_size, out_dim]
  // Perform matrix multiply (M x N) x (N x P) => M x P
  const auto M = weight.size(0); // = out_dim
  const auto N = weight.size(1); // = in_dim

  // Given an N-dimensional input [d0, d1, d2, ..., d_{N-2}, d_{N-1}], the
  // leading dimensions is d0 * d1 * ... * d_{N-2}
  const auto leading_dims = getLeadingDims(src, src.dim() - 1);

  ET_CHECK_MSG(
      out_multiplier.numel() == 1, "out_multiplier should have one element");
  ET_CHECK_MSG(
      out_shift.numel() == 1, "out_multiplier should have one element");

  const int32_t* __restrict__ out_multiplier_data =
      out_multiplier.const_data_ptr<int32_t>();
  const int32_t* __restrict__ out_shift_data =
      out_shift.const_data_ptr<int32_t>();

  // Compute the out_scale from out_multiplier and out_shift
  const float out_scale =
      -out_multiplier_data[0] * 1.0 / (1 << 31) * pow(2, out_shift_data[0]);

  for (int i = 0; i < leading_dims; ++i) {
    for (int j = 0; j < M; ++j) {
      float sum = bias_data[j];
      for (int k = 0; k < N; ++k) {
        sum += (src_data[i * N + k] - src_zero_point) *
            (weight_data[j * N + k] - weight_zero_point);
      }
      out_data[i * M + j] =
          kernels::quantize<T>(sum, out_scale, out_zero_point);
    }
  }
}

void quantized_linear_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    int64_t src_zero_point,
    const Tensor& weight_zero_point_t,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    int64_t out_zero_point,
    __ET_UNUSED const std::optional<Tensor>& offset,
    Tensor& out) {
  // TODO: refactor to use switch case as quantized_linear_per_tensor_out
  if (out.scalar_type() == executorch::aten::ScalarType::Byte) {
    _typed_quantized_linear<uint8_t>(
        src,
        weight,
        bias,
        src_zero_point,
        weight_zero_point_t,
        out_multiplier,
        out_shift,
        out_zero_point,
        out);
  } else if (out.scalar_type() == executorch::aten::ScalarType::Char) {
    _typed_quantized_linear<int8_t>(
        src,
        weight,
        bias,
        src_zero_point,
        weight_zero_point_t,
        out_multiplier,
        out_shift,
        out_zero_point,
        out);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(src.scalar_type()));
  }
}

void quantized_linear_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    __ET_UNUSED const std::optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear_per_tensor(ctype, dtype) \
  case executorch::aten::ScalarType::dtype: {           \
    quantized_linear_per_tensor_<ctype>(                \
        src,                                            \
        weight,                                         \
        bias,                                           \
        src_zero_point,                                 \
        weight_zero_point,                              \
        out_multiplier,                                 \
        out_shift,                                      \
        out_zero_point,                                 \
        out);                                           \
    break;                                              \
  }

  executorch::aten::ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_linear_per_tensor);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", executorch::runtime::toString(dtype));
  }
#undef typed_quantized_linear_per_tensor
}

void quantized_linear_asym8sxasym8s_asym8s_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    __ET_UNUSED const std::optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear_per_tensor(ctype, dtype) \
  case executorch::aten::ScalarType::dtype: {           \
    quantized_linear_per_tensor_<ctype>(                \
        src,                                            \
        weight,                                         \
        bias,                                           \
        src_zero_point,                                 \
        weight_zero_point,                              \
        out_multiplier,                                 \
        out_shift,                                      \
        out_zero_point,                                 \
        out);                                           \
    break;                                              \
  }

  executorch::aten::ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_linear_per_tensor);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", executorch::runtime::toString(dtype));
  }
#undef typed_quantized_linear_per_tensor
}

void quantized_linear_asym8uxasym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    __ET_UNUSED const std::optional<Tensor>& offset,
    Tensor& out) {
#define typed_quantized_linear_per_tensor(ctype, dtype) \
  case executorch::aten::ScalarType::dtype: {           \
    quantized_linear_per_tensor_<ctype>(                \
        src,                                            \
        weight,                                         \
        bias,                                           \
        src_zero_point,                                 \
        weight_zero_point,                              \
        out_multiplier,                                 \
        out_shift,                                      \
        out_zero_point,                                 \
        out);                                           \
    break;                                              \
  }

  executorch::aten::ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_linear_per_tensor);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", executorch::runtime::toString(dtype));
  }
#undef typed_quantized_linear_per_tensor
}

} // namespace native
} // namespace generic
} // namespace impl
