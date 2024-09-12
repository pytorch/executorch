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

  ScalarType val_type = utils::get_scalar_dtype(fill_value);
  ScalarType out_type = out.scalar_type();

  constexpr auto name = "scalar_tensor.out";

  ET_SWITCH_REALB_TYPES(val_type, ctx, name, CTYPE_VAL, [&] {
    CTYPE_VAL val;
    utils::extract_scalar(fill_value, &val);

    ET_SWITCH_REALHB_TYPES(out_type, ctx, name, CTYPE_OUT, [&] {
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
