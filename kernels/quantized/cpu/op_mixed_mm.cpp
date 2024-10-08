/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/vec_ops.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

bool check_quantized_mixed_mm_args(
    const Tensor& in,
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(in, 2));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(weight, 2));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(weight_scales, 1));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(out, 2));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_size_at_dims(in, 1, weight, 0));
  ET_LOG_AND_RETURN_IF_FALSE(
      tensors_have_same_size_at_dims(weight_scales, 0, weight, 0));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, weight_scales, out));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      weight.scalar_type() == ScalarType::Char, "weight dtype must be int8");
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      in.scalar_type() == ScalarType::Float ||
          in.scalar_type() == ScalarType::Half,
      "input dtype must be Float or Half");

  if (opt_weight_zero_points.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_shape(opt_weight_zero_points.value(), weight_scales));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_dtype(opt_weight_zero_points.value(), in));
  }

  // Support for non-null zero points is not implemented yet.
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !opt_weight_zero_points.has_value(), "zero points not supported yet.");
  return true;
}

Tensor& quantized_mixed_mm_out(
    const Tensor& in,
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    Tensor& out) {
  ET_CHECK(check_quantized_mixed_mm_args(
      in, weight, weight_scales, opt_weight_zero_points, out));

  size_t output_ndim = 2;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  output_sizes[0] = in.size(0);
  output_sizes[1] = weight.size(1);

  ET_CHECK(resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok);

  constexpr auto name = "quantized_decomposed::mixed_mm.out";

  ET_SWITCH_TWO_TYPES(Float, Half, in.scalar_type(), ctx, name, CTYPE, [&]() {
    size_t m = in.size(0);
    size_t n = in.size(1);
    size_t p = weight.size(1);

    vec_quantized_matmul_int8<CTYPE>(
        out.mutable_data_ptr<CTYPE>(),
        in.const_data_ptr<CTYPE>(),
        weight.const_data_ptr<int8_t>(),
        weight_scales.const_data_ptr<CTYPE>(),
        m,
        n,
        p);
  });

  return out;
}

Tensor& quantized_mixed_mm_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const Tensor& weight_scales,
    const optional<Tensor>& opt_weight_zero_points,
    Tensor& out) {
  // TODO(mcandales): Remove the need for this wrapper
  (void)ctx;
  return quantized_mixed_mm_out(
      in, weight, weight_scales, opt_weight_zero_points, out);
}

} // namespace native
} // namespace executor
} // namespace torch
