/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& fmod_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       common_type != ScalarType::Bool),
      InvalidArgument,
      out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);
  if (compute_type != ScalarType::Float) {
    compute_type = ScalarType::Double;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "fmod.Tensor_out";

  bool div_by_zero_error = false;

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_bitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::REALHBF16>(
        [&div_by_zero_error](
            const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
          // TODO: rewrite this to be vectorization-capable?
          CTYPE_COMPUTE value = 0;
          if (is_integral_type<CTYPE_COMPUTE, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return value;
            }
          }
          value = std::fmod(val_a, val_b);
          return value;
        },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        b,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  ET_KERNEL_CHECK_MSG(
      ctx,
      !div_by_zero_error,
      InvalidArgument,
      out,
      "Fmod operation encountered integer division by zero");

  return out;
}

Tensor& fmod_Scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = utils::promote_type_with_scalar(a.scalar_type(), b);

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       common_type != ScalarType::Bool),
      InvalidArgument,
      out);

  // Check for intergral division by zero
  ET_KERNEL_CHECK_MSG(
      ctx,
      !(executorch::runtime::isIntegralType(common_type, true) &&
        utils::scalar_to<double>(b) == 0),
      InvalidArgument,
      out,
      "Fmod operation encountered integer division by zero");

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);
  if (compute_type != ScalarType::Float) {
    compute_type = ScalarType::Double;
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "fmod.Scalar_out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
    utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::REALHBF16>(
        [val_b](const auto& val_a) {
          return executorch::math::fmod(val_a, (decltype(val_a))val_b);
        },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
