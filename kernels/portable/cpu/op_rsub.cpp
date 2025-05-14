/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& rsub_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  ScalarType alpha_type = utils::get_scalar_dtype(alpha);

  // Check alpha type
  ET_KERNEL_CHECK(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

  // Common Dtype
  ScalarType common_type = utils::promote_type_with_scalar(a.scalar_type(), b);

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx,
      (common_type == out.scalar_type() && canCast(alpha_type, common_type)),
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
  static constexpr const char op_name[] = "rsub.Scalar_out";

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
    const CTYPE_COMPUTE val_alpha = utils::scalar_to<CTYPE_COMPUTE>(alpha);
    utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::SAME_AS_COMMON>(
        [val_b, val_alpha](const auto val_a) {
          return val_b - val_alpha * val_a;
        },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBF16,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
