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

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

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
  ScalarType min_type = utils::get_scalar_dtype(min);
  ScalarType max_type = utils::get_scalar_dtype(max);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, in_type == out_type, InvalidArgument, out);

  ET_SWITCH_REAL_TYPES(in_type, ctx, "hardtanh.out", CTYPE, [&]() {
    CTYPE min_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(min_type, ctx, "hardtanh.out", CTYPE_MIN, [&]() {
      CTYPE_MIN min_val;
      utils::extract_scalar(min, &min_val);
      min_casted = static_cast<CTYPE>(min_val);
    });

    CTYPE max_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(max_type, ctx, "hardtanh.out", CTYPE_MAX, [&]() {
      CTYPE_MAX max_val;
      utils::extract_scalar(max, &max_val);
      max_casted = static_cast<CTYPE>(max_val);
    });

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
