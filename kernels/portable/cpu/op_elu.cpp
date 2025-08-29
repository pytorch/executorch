/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <type_traits>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch::executor::native {

Tensor& elu_out(
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

  static constexpr const char op_name[] = "elu.out";
  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&]() {
    using MathT = std::
        conditional_t<c10::is_reduced_floating_point_v<CTYPE>, float, CTYPE>;
    const auto math_alpha = utils::scalar_to<MathT>(alpha);
    const auto math_scale = utils::scalar_to<MathT>(scale);
    const auto math_input_scale = utils::scalar_to<MathT>(input_scale);
    const auto negcoef = math_alpha * math_scale;
    utils::apply_unitensor_elementwise_fn<
        CTYPE,
        op_name,
        utils::SupportedTensorDtypes::SAME_AS_COMMON>(
        [negcoef, math_scale, math_input_scale](const CTYPE x) {
          return MathT(x) <= MathT(0)
              ? std::expm1(MathT(x) * math_input_scale) * negcoef
              : MathT(x) * math_scale;
        },
        ctx,
        in,
        utils::SupportedTensorDtypes::FLOATHBF16,
        out);
  });
  return out;
}

} // namespace torch::executor::native
