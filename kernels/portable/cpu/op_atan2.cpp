/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& atan2_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REALHB_TYPES(a_type, ctx, "atan2.out", CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, "atan2.out", CTYPE_B, [&]() {
      ET_SWITCH_FLOATH_TYPES(out_type, ctx, "atan2.out", CTYPE_OUT, [&]() {
        apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
            [](const CTYPE_A val_a, const CTYPE_B val_b) {
              CTYPE_OUT casted_a = static_cast<CTYPE_OUT>(val_a);
              CTYPE_OUT casted_b = static_cast<CTYPE_OUT>(val_b);
              return static_cast<CTYPE_OUT>(std::atan2(casted_a, casted_b));
            },
            a,
            b,
            out);
      });
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
