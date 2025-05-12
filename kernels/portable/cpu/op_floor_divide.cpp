/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

Tensor& floor_divide_out(
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

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "floor_divide.out";

  bool div_by_zero_error = false;

  ET_SWITCH_REAL_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_bitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::REALHBF16>(
        [&div_by_zero_error](
            const CTYPE_COMPUTE val_a, const CTYPE_COMPUTE val_b) {
          // TODO: rewrite this to be vectorization-capable.
          if (is_integral_type<CTYPE_COMPUTE, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return static_cast<CTYPE_COMPUTE>(0);
            }
          }
          return utils::floor_divide(val_a, val_b);
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
      "Floor divide operation encountered integer division by zero");

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
