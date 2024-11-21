/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& sigmoid_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, in.scalar_type() != ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  ET_SWITCH_REALHB_TYPES(in_type, ctx, "sigmoid.out", CTYPE_IN, [&]() {
    ET_SWITCH_FLOATH_TYPES(out_type, ctx, "sigmoid.out", CTYPE_OUT, [&]() {
      apply_unary_map_fn(
          [](const CTYPE_IN val_in) {
            // perform math in double to preserve precision
            double in_casted = static_cast<double>(val_in);
            double out_val = 1.0 / (1.0 + exp(-in_casted));
            return static_cast<CTYPE_OUT>(out_val);
          },
          in.const_data_ptr<CTYPE_IN>(),
          out.mutable_data_ptr<CTYPE_OUT>(),
          in.numel());
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
