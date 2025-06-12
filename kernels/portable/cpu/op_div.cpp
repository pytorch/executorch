/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {

namespace {

ScalarType get_common_type(ScalarType a_type, ScalarType b_type) {
  if (isFloatingType(a_type) && isFloatingType(b_type)) {
    return promoteTypes(a_type, b_type);
  } else if (isFloatingType(a_type)) {
    return a_type;
  } else if (isFloatingType(b_type)) {
    return b_type;
  }
  return ScalarType::Float;
}

} // namespace

Tensor& div_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = get_common_type(a.scalar_type(), b.scalar_type());

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

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_bitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::FLOATHBF16>(
        [](const auto val_a, const auto val_b) { return val_a / val_b; },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        b,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

Tensor& div_out_mode(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    std::optional<std::string_view> mode,
    Tensor& out) {
  if (!mode.has_value()) {
    return div_out(ctx, a, b, out);
  }

  auto mode_val = mode.value();

  // Check mode
  ET_KERNEL_CHECK(
      ctx, mode_val == "trunc" || mode_val == "floor", InvalidArgument, out);

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

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.out_mode";

  const bool mode_is_trunc = mode_val == "trunc";
  bool div_by_zero_error = false;

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_bitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::REALHBF16>(
        [mode_is_trunc, &div_by_zero_error](
            const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
          // TODO: rewrite this to be vectorization-capable.
          if (is_integral_type<CTYPE_COMPUTE, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return static_cast<CTYPE_COMPUTE>(0);
            }
          }
          CTYPE_COMPUTE value = val_a / val_b;
          if (mode_is_trunc) {
            value = std::trunc(value);
          } else {
            // We established above that the mode is either trunc or floor, so
            // it must be floor.
            value = utils::floor_divide(val_a, val_b);
          }
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
      "Div mode operation encountered integer division by zero");

  return out;
}

Tensor& div_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type =
      isFloatingType(a.scalar_type()) ? a.scalar_type() : ScalarType::Float;

  // Check Common Dtype
  ET_KERNEL_CHECK(ctx, common_type == out.scalar_type(), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.Scalar_out";

  ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
    utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::SAME_AS_COMMON>(
        [val_b](const auto val_a) { return val_a / val_b; },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

Tensor& div_scalar_mode_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    std::optional<std::string_view> mode,
    Tensor& out) {
  if (!mode.has_value()) {
    return div_scalar_out(ctx, a, b, out);
  }

  auto mode_val = mode.value();

  // Check mode
  ET_KERNEL_CHECK(
      ctx, mode_val == "trunc" || mode_val == "floor", InvalidArgument, out);

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
      "Div mode operation encountered integer division by zero");

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  const bool mode_is_trunc = mode_val == "trunc";

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.Scalar_mode_out";

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
    utils::apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name>(
        [val_b, mode_is_trunc](const CTYPE_COMPUTE val_a) {
          CTYPE_COMPUTE value = val_a / val_b;
          if (mode_is_trunc) {
            value = std::trunc(value);
          } else {
            value = utils::floor_divide(val_a, val_b);
          }
          return value;
        },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::REALHBF16);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
