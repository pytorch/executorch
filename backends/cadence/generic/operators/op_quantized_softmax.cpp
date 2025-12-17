/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_softmax.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::dequantize;
using ::impl::generic::kernels::quantize;

template <typename T>
void quantized_softmax_per_tensor_(
    const Tensor& input,
    ET_UNUSED const Tensor& mask,
    int64_t dim,
    const float in_scale,
    const int64_t in_zero_point,
    const float out_scale,
    const int64_t out_zero_point,
    Tensor& out) {
  const T* __restrict__ in_data = input.const_data_ptr<T>();
  T* __restrict__ out_data = out.mutable_data_ptr<T>();

  float out_inv_scale = 1.0f / out_scale;
  if (dim < 0) {
    dim += input.dim();
  }
  const int64_t input_size = input.numel();
  float* x = new float[input_size];

  torch::executor::apply_over_dim(
      [in_data,
       out_data,
       x,
       in_scale,
       in_zero_point,
       out_inv_scale,
       out_zero_point](
          const size_t size, const size_t stride, const size_t base) {
        // Dequantize the input tensor
        torch::executor::apply_unary_map_fn(
            [in_scale, in_zero_point](const float val_in) {
              return dequantize<T>(
                  val_in, in_scale, static_cast<int32_t>(in_zero_point));
            },
            in_data + base,
            x + base,
            size,
            stride);

        // Subtract max(X) from input tensor
        float max_in = torch::executor::apply_unary_reduce_fn(
            [](const float val_in, float val_accum) {
              return std::max(val_in, val_accum);
            },
            x + base,
            size,
            stride);

        // Compute exp(X - max(X))
        torch::executor::apply_unary_map_fn(
            [max_in](const float val_in) { return std::exp(val_in - max_in); },
            x + base,
            x + base,
            size,
            stride);

        // Compute sum(exp(X - max(X))
        float temp_sum = torch::executor::apply_unary_reduce_fn(
            [](const float val_in, float val_accum) {
              return val_accum + val_in;
            },
            x + base,
            size,
            stride);

        // Compute exp(X - max(X)) / sum(exp(X - max(X)) and quantize the
        float recip = 1.0 / temp_sum;
        torch::executor::apply_unary_map_fn(
            [recip, out_inv_scale, out_zero_point](const float val_in) {
              float res = val_in * recip;
              return quantize<T>(
                  res, out_inv_scale, static_cast<int32_t>(out_zero_point));
            },
            x + base,
            out_data + base,
            size,
            stride);
      },
      input,
      dim);

  delete[] x;
}

// Compute quantized softmax. The current implementation assumes that the
// input is per-tensor quantized.
template <typename T>
void quantized_softmax_(
    const Tensor& input,
    const Tensor& mask,
    const int64_t dim,
    const Tensor& in_scale,
    const Tensor& in_zero_point,
    const Tensor& out_scale,
    const Tensor& out_zero_point,
    Tensor& out) {
  // Extract the zero point and scale for input tensor.
  float input_scale = in_scale.const_data_ptr<float>()[0];
  int64_t input_zero_point = in_zero_point.const_data_ptr<int64_t>()[0];
  float output_scale = out_scale.const_data_ptr<float>()[0];
  int64_t output_zero_point = out_zero_point.const_data_ptr<int64_t>()[0];
  quantized_softmax_per_tensor_<T>(
      input,
      mask,
      dim,
      input_scale,
      input_zero_point,
      output_scale,
      output_zero_point,
      out);
}

} // namespace

Tensor& quantized_softmax_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& mask,
    int64_t dim,
    const Tensor& in_scale,
    const Tensor& in_zero_point,
    const Tensor& out_scale,
    const Tensor& out_zero_point,
    Tensor& out) {
#define typed_quantized_softmax(ctype, dtype) \
  case ScalarType::dtype: {                   \
    quantized_softmax_<ctype>(                \
        input,                                \
        mask,                                 \
        dim,                                  \
        in_scale,                             \
        in_zero_point,                        \
        out_scale,                            \
        out_zero_point,                       \
        out);                                 \
    break;                                    \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_softmax)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_softmax
  return out;
}

Tensor& quantized_softmax_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& mask,
    int64_t dim,
    double in_scale,
    int64_t in_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
#define typed_quantized_softmax(ctype, dtype) \
  case ScalarType::dtype: {                   \
    quantized_softmax_per_tensor_<ctype>(     \
        input,                                \
        mask,                                 \
        dim,                                  \
        in_scale,                             \
        in_zero_point,                        \
        out_scale,                            \
        out_zero_point,                       \
        out);                                 \
    break;                                    \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_softmax)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_softmax
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
