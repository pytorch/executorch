/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& full_like_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Scalar& fill_value,
    optional<MemoryFormat> memory_format,
    Tensor& out) {
  (void)ctx;

  if (memory_format.has_value()) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        memory_format.value() == MemoryFormat::Contiguous,
        InvalidArgument,
        out,
        "memory_format must be contiguous");
  }

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType val_type = utils::get_scalar_dtype(fill_value);
  ScalarType out_type = out.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(
      Bool, val_type, ctx, "full_like.out", CTYPE_VAL, [&] {
        CTYPE_VAL val;
        utils::extract_scalar(fill_value, &val);

        ET_SWITCH_REAL_TYPES_AND(
            Bool, out_type, ctx, "full_like.out", CTYPE_OUT, [&] {
              CTYPE_OUT val_casted = static_cast<CTYPE_OUT>(val);
              auto data_out = out.mutable_data_ptr<CTYPE_OUT>();
              for (size_t i = 0; i < out.numel(); ++i) {
                data_out[i] = val_casted;
              }
            });
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
