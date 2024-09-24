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
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::Tensor;

namespace {

// Rounds a floating point value to the closest integer. Values with a
// fractional part of exactly 0.5 are rounded to the closest even integer. Uses
// the implementation from torch/src/jit/runtime/register_ops_utils.h.
template <typename CTYPE>
inline CTYPE round_to_even(CTYPE a) {
  return a - std::floor(a) == 0.5 ? (std::round(a * 0.5) * 2.0) : std::round(a);
}

} // namespace

Tensor& round_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_shape_and_dtype(in, out), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_real_type(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  auto in_scalar_type = in.scalar_type();

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "round.out", CTYPE, [&] {
    apply_unary_map_fn(
        [in_scalar_type](const CTYPE val_in) {
          if (isIntegralType(in_scalar_type, /*includeBool=*/false)) {
            return val_in;
          } else {
            return static_cast<CTYPE>(round_to_even<CTYPE>(val_in));
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
