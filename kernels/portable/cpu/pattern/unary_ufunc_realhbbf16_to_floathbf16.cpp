/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

Tensor& unary_ufunc_realhbbf16_to_floathbf16(
    double (*fn)(double),
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(out), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  const auto in_type = in.scalar_type();
  const auto out_type = out.scalar_type();

  // TODO: this is broken for dtype_selective_build: this was
  // __func__, which isn't the operator name.
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] =
      "unary_ufunc_realhbbf16_to_floathbf16";

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, op_name, CTYPE_IN, [&] {
    utils::apply_unitensor_elementwise_fn<CTYPE_IN, op_name>(
        [fn](const CTYPE_IN val_in) { return fn(val_in); },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        out,
        utils::SupportedTensorDtypes::FLOATHBF16);
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
