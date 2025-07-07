/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& hardtanh_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Scalar& min,
    const Scalar& max,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, in_type == out_type, InvalidArgument, out);

  ET_SWITCH_REALHBF16_TYPES(in_type, ctx, "hardtanh.out", CTYPE, [&]() {
    auto opt_min_casted =
        utils::internal::check_overflow_scalar_cast<CTYPE>(min);
    ET_KERNEL_CHECK(ctx, opt_min_casted.has_value(), InvalidArgument, );
    auto min_casted = opt_min_casted.value();

    auto opt_max_casted =
        utils::internal::check_overflow_scalar_cast<CTYPE>(max);
    ET_KERNEL_CHECK(ctx, opt_max_casted.has_value(), InvalidArgument, );
    auto max_casted = opt_max_casted.value();

    apply_unary_map_fn(
        [min_casted, max_casted](const CTYPE val_in) {
          return utils::min_override(
              utils::max_override(val_in, min_casted), max_casted);
        },
        in.const_data_ptr<CTYPE>(),
        out.mutable_data_ptr<CTYPE>(),
        in.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
