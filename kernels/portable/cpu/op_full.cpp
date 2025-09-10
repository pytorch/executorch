/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& full_out(
    KernelRuntimeContext& ctx,
    const IntArrayRef sizes,
    const Scalar& fill_value,
    Tensor& out) {
  (void)ctx;

  ScalarType out_type = out.scalar_type();

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, sizes) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "full.out";

  ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, op_name, CTYPE_OUT, [&] {
    auto opt_val_casted =
        utils::internal::check_overflow_scalar_cast<CTYPE_OUT>(fill_value);
    ET_KERNEL_CHECK(ctx, opt_val_casted.has_value(), InvalidArgument, );
    auto val_casted = opt_val_casted.value();
    auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
    for (const auto i : c10::irange(out.numel())) {
      data_out[i] = val_casted;
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
