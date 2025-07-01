/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

Tensor& add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out.scalar_type()) &&
       check_alpha_type(utils::get_scalar_dtype(alpha), common_type)),
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
  static constexpr const char op_name[] = "add.out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    CTYPE_COMPUTE val_alpha;
    ET_KERNEL_CHECK(
        ctx, utils::extract_scalar(alpha, &val_alpha), InvalidArgument, );
    utils::apply_bitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::REALHBBF16>(
        [val_alpha](const auto val_a, const auto val_b) {
          return val_a + val_alpha * val_b;
        },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        b,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

Tensor& add_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = utils::promote_type_with_scalar(a.scalar_type(), b);

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (common_type == out.scalar_type() &&
       check_alpha_type(utils::get_scalar_dtype(alpha), common_type)),
      InvalidArgument,
      out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "add.Scalar_out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
    CTYPE_COMPUTE val_alpha;
    ET_KERNEL_CHECK(
        ctx, utils::extract_scalar(alpha, &val_alpha), InvalidArgument, );
    auto val_alpha_times_b = val_alpha * val_b;
    utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::SAME_AS_COMMON>(
        [val_alpha_times_b](const auto val_a) {
          // Cast here supports vectorization; either it does nothing
          // or it casts from CTYPE_COMPUTE to
          // Vectorized<CTYPE_COMPUTE>.
          return val_a + decltype(val_a)(val_alpha_times_b);
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
