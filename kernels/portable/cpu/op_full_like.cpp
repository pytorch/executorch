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

Tensor& full_like_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Scalar& fill_value,
    optional<MemoryFormat> memory_format,
    Tensor& out) {
  (void)ctx;

  if (memory_format.has_value()) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        memory_format.value() == MemoryFormat::Contiguous ||
            memory_format.value() == MemoryFormat::Preserve,
        InvalidArgument,
        out,
        "memory_format must be contiguous");
  }

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType out_type = out.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "full_like.out";

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
