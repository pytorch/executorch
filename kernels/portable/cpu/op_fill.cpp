/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Scalar = executorch::aten::Scalar;
using ScalarType = executorch::aten::ScalarType;
using Tensor = executorch::aten::Tensor;

Tensor& fill_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, a_type == out_type, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "fill.Scalar_out";

  ET_SWITCH_REALHBBF16_TYPES(a_type, ctx, op_name, CTYPE_A, [&] {
    auto opt_b_casted = utils::internal::check_overflow_scalar_cast<CTYPE_A>(b);
    ET_KERNEL_CHECK(ctx, opt_b_casted.has_value(), InvalidArgument, );
    auto b_casted = opt_b_casted.value();

    apply_unary_map_fn(
        [b_casted](const CTYPE_A val_a) { return b_casted; },
        a.const_data_ptr<CTYPE_A>(),
        out.mutable_data_ptr<CTYPE_A>(),
        out.numel());
  });

  return out;
}

Tensor& fill_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Assert `b` must be a scalar tensor.
  ET_KERNEL_CHECK(ctx, tensor_is_scalar(b), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, a_type == out_type, InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "fill.Tensor_out";

  ET_SWITCH_REALHBBF16_TYPES(a_type, ctx, op_name, CTYPE_A, [&] {
    CTYPE_A b_casted{};
    ET_SWITCH_REALHBBF16_TYPES(b_type, ctx, op_name, CTYPE_B, [&] {
      CTYPE_B b_val;
      ET_EXTRACT_SCALAR_TENSOR(b, b_val);
      b_casted = static_cast<CTYPE_A>(b_val);
    });

    apply_unary_map_fn(
        [b_casted](const CTYPE_A val_a) { return b_casted; },
        a.const_data_ptr<CTYPE_A>(),
        out.mutable_data_ptr<CTYPE_A>(),
        out.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
