/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& where_out(
    KernelRuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Common Dtype
  ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

  // Check Common Dtype
  ET_KERNEL_CHECK(ctx, common_type == out.scalar_type(), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(cond, a, b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, cond, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "where.self_out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_tritensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::SAME_AS_COMMON>(
        [](const CTYPE_COMPUTE val_a,
           const CTYPE_COMPUTE val_b,
           const CTYPE_COMPUTE val_c) { return val_c ? val_a : val_b; },
        ctx,
        a,
        utils::SupportedTensorDtypes::REALHBBF16,
        b,
        utils::SupportedTensorDtypes::REALHBBF16,
        cond,
        utils::SupportedTensorDtypes::BOOL_OR_BYTE,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
