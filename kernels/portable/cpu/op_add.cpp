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
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      (executorch::runtime::tensor_is_realhbbf16_type(a) &&
       executorch::runtime::tensor_is_realhbbf16_type(b) &&
       executorch::runtime::tensor_is_realhbbf16_type(out)),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType alpha_type = utils::get_scalar_dtype(alpha);
  ScalarType common_type = promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, check_alpha_type(alpha_type, common_type), InvalidArgument, out);

  static constexpr const char op_name[] = "add.out";

  ET_SWITCH_REALB_TYPES(common_type, ctx, op_name, CTYPE_COMMON, [&]() {
    utils::apply_bitensor_elementwise_fn<CTYPE_COMMON, op_name>(
        [alpha](const CTYPE_COMMON val_a, const CTYPE_COMMON val_b) {
          CTYPE_COMMON val_alpha = utils::scalar_to<CTYPE_COMMON>(alpha);
          return val_a + val_alpha * val_b;
        },
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        b,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

Tensor& add_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx,
      (executorch::runtime::tensor_is_realhbbf16_type(a) &&
       executorch::runtime::tensor_is_realhbbf16_type(out)),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType alpha_type = utils::get_scalar_dtype(alpha);
  ScalarType common_type =
      utils::promote_type_with_scalar(a_type, b, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, check_alpha_type(alpha_type, common_type), InvalidArgument, out);

  if (common_type == ScalarType::Half || common_type == ScalarType::BFloat16) {
    common_type = ScalarType::Float;
  }

  static constexpr const char op_name[] = "add.Scalar_out";

  ET_SWITCH_REALB_TYPES(common_type, ctx, op_name, CTYPE_COMMON, [&]() {
    utils::apply_unitensor_elementwise_fn<CTYPE_COMMON, op_name>(
        [b, alpha](const CTYPE_COMMON val_a) {
          CTYPE_COMMON val_b = utils::scalar_to<CTYPE_COMMON>(b);
          CTYPE_COMMON val_alpha = utils::scalar_to<CTYPE_COMMON>(alpha);
          return val_a + val_alpha * val_b;
        },
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::REALHBBF16);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
