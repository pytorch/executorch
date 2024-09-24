/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cinttypes>
#include <cmath>
#include <cstdint>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

double exp_overload(double d) {
  return exp(d);
}

float exp_overload(float f) {
  return expf(f);
}

/**
 * In-place element-wise sigmoid function , i.e., f(x) = 1 / (1 + e^{-x})
 */
// TODO: T146333648, refactor this as a common helper function
template <typename CTYPE_OUT>
void sigmoid_tensor(Tensor& out) {
  CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
  for (size_t i = 0; i < out.numel(); i++) {
    out_data[i] = 1.0 / (1.0 + exp_overload(-out_data[i]));
  }
}

/**
 * Element-wise multiplication of the first half of `in` along the specified
 * dimension and `out`, overwriting `out`.
 */
template <typename CTYPE_IN, typename CTYPE_OUT>
void mul_tensors(const Tensor& in, int64_t dim, Tensor& out) {
  size_t num_values = static_cast<size_t>(in.size(dim)) / 2;
  size_t dim_length_in = static_cast<size_t>(in.size(dim));
  size_t dim_length_out = static_cast<size_t>(out.size(dim));
  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);

  const CTYPE_IN* input_data_base = in.const_data_ptr<CTYPE_IN>();
  CTYPE_OUT* output_data_base = out.mutable_data_ptr<CTYPE_OUT>();

  for (size_t i = 0; i < leading_dims; i++) {
    const CTYPE_IN* input_data =
        input_data_base + i * dim_length_in * trailing_dims;
    CTYPE_OUT* output_data =
        output_data_base + i * dim_length_out * trailing_dims;
    for (size_t j = 0; j < num_values; j++) {
      for (size_t k = 0; k < trailing_dims; ++k) {
        output_data[k] = static_cast<CTYPE_OUT>(input_data[k]) * output_data[k];
      }
      input_data += trailing_dims;
      output_data += trailing_dims;
    }
  }
}

/**
 * Slice the tensor in the given dim, from start to end, assume tensor in and
 * out have same shape and dtype, the dim is a non-negative number and start,
 * end are valid non-negative number
 */
template <typename CTYPE_IN, typename CTYPE_OUT>
void slice_tensor(
    const Tensor& in,
    int64_t dim,
    int64_t start,
    int64_t end,
    Tensor& out) {
  size_t num_values = static_cast<size_t>(end - start);
  size_t dim_length_in = static_cast<size_t>(in.size(dim));
  size_t dim_length_out = static_cast<size_t>(out.size(dim));
  size_t non_negative_start = static_cast<size_t>(start);
  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_dims = getTrailingDims(in, dim);

  const CTYPE_IN* input_data_base = in.const_data_ptr<CTYPE_IN>();
  CTYPE_OUT* output_data_base = out.mutable_data_ptr<CTYPE_OUT>();

  for (size_t i = 0; i < leading_dims; i++) {
    const CTYPE_IN* input_data = input_data_base +
        (i * dim_length_in + non_negative_start) * trailing_dims;
    CTYPE_OUT* output_data =
        output_data_base + i * dim_length_out * trailing_dims;
    for (size_t j = 0; j < num_values; j++) {
      for (size_t k = 0; k < trailing_dims; ++k) {
        output_data[k] = static_cast<CTYPE_OUT>(input_data[k]);
      }
      input_data += trailing_dims;
      output_data += trailing_dims;
    }
  }
}

/**
 * Applies the gated linear unit function
 *
 * Based on the characteristic of glu function, the output should be in
 * floating point type (Float and Double). The input and output tensors don't
 * necessarily need to have the same type. Here are the assertions:
 *  1. The input shall be in any float types (Float, Double)
 *  2. The output shall be in float types (Float, Double)
 */
template <typename CTYPE_IN, typename CTYPE_OUT>
Tensor& glu_out_tensor(const Tensor& self, int64_t dim, Tensor& out) {
  const auto self_size = self.size(dim);
  slice_tensor<CTYPE_IN, CTYPE_OUT>(self, dim, self_size / 2, self_size, out);
  sigmoid_tensor<CTYPE_OUT>(out);
  mul_tensors<CTYPE_IN, CTYPE_OUT>(self, dim, out);
  return out;
}
} // namespace

/**
 * Applies the gated linear unit function
 *
 * Based on the characteristic of glu function, the output should be in
 * floating point type (Float and Double). The input and output tensors don't
 * necessarily need to have the same type. Here are the assertions:
 *  1. The input shall be in any float types (Float, Double)
 *  2. The output shall be in float types (Float, Double)
 */
Tensor& glu_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_glu_out(self, dim, out) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, check_glu_args(self, dim, out), InvalidArgument, out);

  const size_t non_negative_dim = dim < 0 ? dim + self.dim() : dim;
  const auto in_dtype = self.scalar_type();

  ET_SWITCH_FLOAT_TYPES(in_dtype, ctx, "glu", CTYPE_IN, [&]() {
    if (out.scalar_type() == ScalarType::Float) {
      glu_out_tensor<CTYPE_IN, float>(self, non_negative_dim, out);
    } else {
      glu_out_tensor<CTYPE_IN, double>(self, non_negative_dim, out);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
