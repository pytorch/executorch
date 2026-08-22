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

Tensor& unary_ufunc_realhbbf16_to_bool(
    bool (*fn_float)(float),
    bool (*fn_double)(double),
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

  ET_KERNEL_CHECK_MSG(
      ctx,
      out.scalar_type() == executorch::aten::ScalarType::Bool,
      InvalidArgument,
      out,
      "Expected out tensor to have dtype Bool, but got %" PRId8 " instead.",
      static_cast<int8_t>(out.scalar_type()));

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  const auto in_type = in.scalar_type();

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, __func__, CTYPE_IN, [&] {
    apply_unary_map_fn(
        [fn_double, fn_float](const CTYPE_IN val_in) {
          if constexpr (std::is_same_v<CTYPE_IN, double>) {
            (void)fn_float;
            double xi = static_cast<double>(val_in);
            return static_cast<bool>(fn_double(xi));
          } else {
            (void)fn_double;
            float xi = static_cast<float>(val_in);
            return static_cast<bool>(fn_float(xi));
          }
        },
        in.const_data_ptr<CTYPE_IN>(),
        out.mutable_data_ptr<bool>(),
        in.numel());
  });

  return out;
}

} // namespace internal
} // namespace native
} // namespace executor
} // namespace torch
