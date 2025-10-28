/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch::executor::native {

namespace {
template <typename CTYPE>
void elu(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& out) {
  const CTYPE* in_data = input.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  using MathT =
      std::conditional_t<c10::is_reduced_floating_point_v<CTYPE>, float, CTYPE>;
  const auto math_alpha = utils::scalar_to<MathT>(alpha);
  const auto math_scale = utils::scalar_to<MathT>(scale);
  const auto math_input_scale = utils::scalar_to<MathT>(input_scale);

  using Vec = at::vec::Vectorized<CTYPE>;
  at::vec::map(
      [math_alpha, math_scale, math_input_scale](Vec x) {
        auto scaled_input = x * Vec(static_cast<CTYPE>(math_input_scale));
        auto zero = Vec(static_cast<CTYPE>(0));
        auto one = Vec(static_cast<CTYPE>(1));
        auto alpha_vec = Vec(static_cast<CTYPE>(math_alpha));
        auto scale_vec = Vec(static_cast<CTYPE>(math_scale));

        auto pos_mask = scaled_input > zero;
        auto neg_result = alpha_vec * ((scaled_input.exp()) - one);
        auto result = Vec::blendv(neg_result, scaled_input, pos_mask);
        return result * scale_vec;
      },
      out_data,
      in_data,
      out.numel());
}
} // namespace

Tensor& opt_elu_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(in), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, "elu.out", CTYPE, [&]() {
    elu<CTYPE>(ctx, in, alpha, scale, input_scale, out);
  });
  return out;
}

} // namespace torch::executor::native
