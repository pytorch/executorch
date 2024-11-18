/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/pattern/pattern.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace internal {

Tensor& unary_ufunc_realh(
    double (*fn)(double),
    KernelRuntimeContext& ctx,
    const Tensor& in,
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
      ctx, tensors_have_same_shape_and_dtype(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_SWITCH_REALH_TYPES(in.scalar_type(), ctx, __func__, CTYPE, [&] {
    apply_unary_map_fn(
        [fn](const CTYPE val_in) { return static_cast<CTYPE>(fn(val_in)); },
        in.const_data_ptr<CTYPE>(),
        out.mutable_data_ptr<CTYPE>(),
        in.numel());
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
