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

Tensor& unary_ufunc_realhbf16(
    float (*fn_float)(float),
    double (*fn_double)(double),
    KernelRuntimeContext& ctx,
    const Tensor& in,
    Tensor& out) {
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

  ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, __func__, CTYPE, [&] {
    apply_unary_map_fn(
        [fn_double, fn_float](const CTYPE val_in) {
          if constexpr (std::is_same_v<CTYPE, double>) {
            (void)fn_float;
            double xi = static_cast<double>(val_in);
            return fn_double(xi);
          } else {
            (void)fn_double;
            float xi = static_cast<float>(val_in);
            return static_cast<CTYPE>(fn_float(xi));
          }
        },
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
