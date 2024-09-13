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

using exec_aten::Tensor;

/**
 * Computes the bitwise NOT of the given input tensor. The input tensor must be
 * of Integral or Boolean types. For bool tensors, it computes the logical NOT.
 **/
Tensor&
bitwise_not_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  if (in.scalar_type() == exec_aten::ScalarType::Bool) {
    apply_unary_map_fn(
        [](const bool val_in) { return !val_in; },
        in.const_data_ptr<bool>(),
        out.mutable_data_ptr<bool>(),
        in.numel());
  } else if (isIntegralType(in.scalar_type(), /*includeBool=*/false)) {
    ET_SWITCH_INT_TYPES(in.scalar_type(), ctx, "bitwise_not.out", CTYPE, [&] {
      apply_unary_map_fn(
          [](const CTYPE val_in) { return ~val_in; },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    });
  } else {
    ET_KERNEL_CHECK_MSG(
        ctx,
        false,
        InvalidArgument,
        out,
        "Unsupported input dtype %" PRId8,
        static_cast<int8_t>(in.scalar_type()));
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
