/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::Tensor;

Tensor& abs_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  const bool in_is_complex =
      executorch::runtime::isComplexType(in.scalar_type());
  ET_KERNEL_CHECK(
      ctx,
      in_is_complex || tensors_have_same_dtype(in, out),
      InvalidArgument,
      out);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "abs.out";

  if (in_is_complex) {
    // NOTE: Elected not to add COMPLEXH to dtype_util.h for now
    // because I am not planning wide rollout of complex support; if
    // we do add SupportedTensorDtypes::COMPLEXH support, then we
    // should use it here.
    ET_SWITCH_COMPLEXH_TYPES(in.scalar_type(), ctx, op_name, CTYPE_IN, [&] {
      ET_SWITCH_FLOATH_TYPES(out.scalar_type(), ctx, op_name, CTYPE_OUT, [&] {
        apply_unary_map_fn<CTYPE_IN, CTYPE_OUT>(
            [](const CTYPE_IN val_in) -> CTYPE_OUT {
              return sqrt(
                  val_in.real_ * val_in.real_ + val_in.imag_ * val_in.imag_);
            },
            in.const_data_ptr<CTYPE_IN>(),
            out.mutable_data_ptr<CTYPE_OUT>(),
            in.numel());
      });
    });
  } else {
    ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
      apply_unary_map_fn(
          [](const CTYPE val_in) {
            if (val_in < 0) {
              return static_cast<CTYPE>(-val_in);
            } else {
              return static_cast<CTYPE>(val_in);
            }
          },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
