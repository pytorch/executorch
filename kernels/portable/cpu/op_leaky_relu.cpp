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
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& leaky_relu_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Scalar& negative_slope,
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

  ET_SWITCH_FLOATHBF16_TYPES(in_type, ctx, "leaky_relu.out", CTYPE, [&]() {
    auto opt_negative_slope_casted =
        utils::internal::check_overflow_scalar_cast<CTYPE>(negative_slope);
    ET_KERNEL_CHECK(
        ctx, opt_negative_slope_casted.has_value(), InvalidArgument, );
    auto negative_slope_casted = opt_negative_slope_casted.value();

    apply_unary_map_fn(
        [negative_slope_casted](const CTYPE val_in) {
          if (val_in >= 0) {
            return val_in;
          } else {
            return val_in * negative_slope_casted;
          }
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
